/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*
 * The code was taken from Mozilla DeepSpeech project:
 * https://github.com/mozilla/DeepSpeech/tree/master/native_client
 */


#include <algorithm>
#include <vector>
#include <cmath>

#include "beam_search.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/kernels/bounds_check.h"


namespace tensorflow {
namespace ctc {


template <typename CTCBeamState = ctc_beam_search::EmptyBeamState,
          typename CTCBeamComparer =
              ctc_beam_search::BeamComparer<CTCBeamState>>
class CTCBeamSearchNormLogDecoder : public CTCDecoder {
  // Beam Search
  //
  // Example (GravesTh Fig. 7.5):
  //         a    -
  //  P = [ 0.3  0.7 ]  t = 0
  //      [ 0.4  0.6 ]  t = 1
  //
  // Then P(l = -) = P(--) = 0.7 * 0.6 = 0.42
  //      P(l = a) = P(a-) + P(aa) + P(-a) = 0.3*0.4 + ... = 0.58
  //
  // In this case, Best Path decoding is suboptimal.
  //
  // For Beam Search, we use the following main recurrence relations:
  //
  // Relation 1:
  // ---------------------------------------------------------- Eq. 1
  //      P(l=abcd @ t=7) = P(l=abc  @ t=6) * P(d @ 7)
  //                      + P(l=abcd @ t=6) * (P(d @ 7) + P(- @ 7))
  // where P(l=? @ t=7), ? = a, ab, abc, abcd are all stored and
  // updated recursively in the beam entry.
  //
  // Relation 2:
  // ---------------------------------------------------------- Eq. 2
  //      P(l=abc? @ t=3) = P(l=abc @ t=2) * P(? @ 3)
  // for ? in a, b, d, ..., (not including c or the blank index),
  // and the recurrence starts from the beam entry for P(l=abc @ t=2).
  //
  // For this case, the length of the new sequence equals t+1 (t
  // starts at 0).  This special case can be calculated as:
  //   P(l=abc? @ t=3) = P(a @ 0)*P(b @ 1)*P(c @ 2)*P(? @ 3)
  // but we calculate it recursively for speed purposes.
  typedef ctc_beam_search::BeamEntry<CTCBeamState> BeamEntry;
  typedef ctc_beam_search::BeamRoot<CTCBeamState> BeamRoot;
  typedef ctc_beam_search::BeamProbability BeamProbability;

 public:
  typedef BaseBeamScorer<CTCBeamState> DefaultBeamScorer;

  // The beam search decoder is constructed specifying the beam_width (number of
  // candidates to keep at each decoding timestep) and a beam scorer (used for
  // custom scoring, for example enabling the use of a language model).
  // The ownership of the scorer remains with the caller. The default
  // implementation, CTCBeamSearchDecoder<>::DefaultBeamScorer, generates the
  // standard beam search.
  CTCBeamSearchNormLogDecoder(int num_classes, int beam_width,
                       BaseBeamScorer<CTCBeamState>* scorer, int batch_size = 1,
                       bool merge_repeated = false)
      : CTCDecoder(num_classes, batch_size, merge_repeated),
        beam_width_(beam_width),
        leaves_(beam_width),
        beam_scorer_(CHECK_NOTNULL(scorer)) {
    Reset();
  }

  ~CTCBeamSearchNormLogDecoder() override {}

  // Run the hibernating beam search algorithm on the given input.
  Status Decode(const CTCDecoder::SequenceLength& seq_len,
                const std::vector<CTCDecoder::Input>& input,
                std::vector<CTCDecoder::Output>* output,
                CTCDecoder::ScoreOutput* scores) override;

  // Calculate the next step of the beam search and update the internal state.
  template <typename Vector>
  void Step(const Vector& log_input_t);

  template <typename Vector>
  float GetTopK(const int K, const Vector& input,
                std::vector<float>* top_k_logits,
                std::vector<int>* top_k_indices);

  // Retrieve the beam scorer instance used during decoding.
  BaseBeamScorer<CTCBeamState>* GetBeamScorer() const { return beam_scorer_; }

  // Set label selection parameters for faster decoding.
  // See comments for label_selection_size_ and label_selection_margin_.
  void SetLabelSelectionParameters(int label_selection_size,
                                   float label_selection_margin) {
    label_selection_size_ = label_selection_size;
    label_selection_margin_ = label_selection_margin;
  }

  // Reset the beam search
  void Reset();

  // Extract the top n paths at current time step
  Status TopPaths(int n, std::vector<std::vector<int>>* paths,
                  std::vector<float>* log_probs, bool merge_repeated) const;

  gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
  BaseBeamScorer<CTCBeamState>* beam_scorer_;

 private:
  int beam_width_;

  // Label selection is designed to avoid possibly very expensive scorer calls,
  // by pruning the hypotheses based on the input alone.
  // Label selection size controls how many items in each beam are passed
  // through to the beam scorer. Only items with top N input scores are
  // considered.
  // Label selection margin controls the difference between minimal input score
  // (versus the best scoring label) for an item to be passed to the beam
  // scorer. This margin is expressed in terms of log-probability.
  // Default is to do no label selection.
  // For more detail: https://research.google.com/pubs/pub44823.html
  int label_selection_size_ = 0;       // zero means unlimited
  float label_selection_margin_ = -1;  // -1 means unlimited.

  // gtl::TopN<BeamEntry*, CTCBeamComparer> leaves_;
  std::unique_ptr<BeamRoot> beam_root_;
  // BaseBeamScorer<CTCBeamState>* beam_scorer_;

  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchNormLogDecoder);
};

template <typename CTCBeamState, typename CTCBeamComparer>
Status CTCBeamSearchNormLogDecoder<CTCBeamState, CTCBeamComparer>::Decode(
    const CTCDecoder::SequenceLength& seq_len,
    const std::vector<CTCDecoder::Input>& input,
    std::vector<CTCDecoder::Output>* output, ScoreOutput* scores) {
  // Storage for top paths.
  std::vector<std::vector<int>> beams;
  std::vector<float> beam_log_probabilities;
  int top_n = output->size();
  if (std::any_of(output->begin(), output->end(),
                  [this](const CTCDecoder::Output& output) -> bool {
                    return output.size() < this->batch_size_;
                  })) {
    return errors::InvalidArgument(
        "output needs to be of size at least (top_n, batch_size).");
  }
  if (scores->rows() < batch_size_ || scores->cols() < top_n) {
    return errors::InvalidArgument(
        "scores needs to be of size at least (batch_size, top_n).");
  }

  for (int b = 0; b < batch_size_; ++b) {
    int seq_len_b = seq_len[b];
    Reset();

    for (int t = 0; t < seq_len_b; ++t) {
      // Pass log-probabilities for this example + time.
      Step(input[t].row(b));
    }  // for (int t...

    // O(n * log(n))
    std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
    leaves_.Reset();
    for (int i = 0; i < branches->size(); ++i) {
      BeamEntry* entry = (*branches)[i];
      beam_scorer_->ExpandStateEnd(&entry->state);
      entry->newp.total +=
          beam_scorer_->GetStateEndExpansionScore(entry->state);
      leaves_.push(entry);
    }

    Status status =
        TopPaths(top_n, &beams, &beam_log_probabilities, merge_repeated_);
    if (!status.ok()) {
      return status;
    }

    CHECK_EQ(top_n, beam_log_probabilities.size());
    CHECK_EQ(beams.size(), beam_log_probabilities.size());

    for (int i = 0; i < top_n; ++i) {
      // Copy output to the correct beam + batch
      (*output)[i][b].swap(beams[i]);
      (*scores)(b, i) = -beam_log_probabilities[i];
    }
  }  // for (int b...
  return Status::OK();
}

template <typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
float CTCBeamSearchNormLogDecoder<CTCBeamState, CTCBeamComparer>::GetTopK(
    const int K, const Vector& input, std::vector<float>* top_k_logits,
    std::vector<int>* top_k_indices) {
  // Find Top K choices, complexity nk in worst case. The array input is read
  // just once.
  CHECK_EQ(num_classes_, input.size());
  top_k_logits->clear();
  top_k_indices->clear();
  top_k_logits->resize(K, -INFINITY);
  top_k_indices->resize(K, -1);
  for (int j = 0; j < num_classes_ - 1; ++j) {
    const float logit = input(j);
    if (logit > (*top_k_logits)[K - 1]) {
      int k = K - 1;
      while (k > 0 && logit > (*top_k_logits)[k - 1]) {
        (*top_k_logits)[k] = (*top_k_logits)[k - 1];
        (*top_k_indices)[k] = (*top_k_indices)[k - 1];
        k--;
      }
      (*top_k_logits)[k] = logit;
      (*top_k_indices)[k] = j;
    }
  }
  // Return max value which is in 0th index or blank character logit
  return std::max((*top_k_logits)[0], input(num_classes_ - 1));
}

template <typename CTCBeamState, typename CTCBeamComparer>
template <typename Vector>
void CTCBeamSearchNormLogDecoder<CTCBeamState, CTCBeamComparer>::Step(
    const Vector& raw_input) {
  std::vector<float> top_k_logits;
  std::vector<int> top_k_indices;
  const bool top_k =
      (label_selection_size_ > 0 && label_selection_size_ < raw_input.size());
  // Number of character classes to consider in each step.
  const int max_classes = top_k ? label_selection_size_ : (num_classes_ - 1);
  // Get max coefficient and remove it from raw_input later.
  float max_coeff;
  if (top_k) {
    max_coeff = GetTopK(label_selection_size_, raw_input, &top_k_logits,
                        &top_k_indices);
  } else {
    max_coeff = raw_input.maxCoeff();
  }
  // Get normalization term of softmax: log(sum(exp(logit[j]-max_coeff))).
  float logsumexp = 0.0;
  for (int j = 0; j < raw_input.size(); ++j) {
    logsumexp += expf(raw_input(j) - max_coeff);
  }
  logsumexp = logf(logsumexp);

  // Final normalization offset to get correct log probabilities.
  float norm_offset = max_coeff + logsumexp;

  const float label_selection_input_min =
      (label_selection_margin_ >= 0) ? (max_coeff - label_selection_margin_)
                                     : -std::numeric_limits<float>::infinity();

  // Extract the beams sorted in decreasing new probability
  CHECK_EQ(num_classes_, raw_input.size());

  std::unique_ptr<std::vector<BeamEntry*>> branches(leaves_.Extract());
  leaves_.Reset();

  for (BeamEntry* b : *branches) {
    // P(.. @ t) becomes the new P(.. @ t-1)
    b->oldp = b->newp;
  }

  for (BeamEntry* b : *branches) {
    if (b->parent != nullptr) {  // if not the root
      if (b->parent->Active()) {
        // If last two sequence characters are identical:
        //   Plabel(l=acc @ t=6) = (Plabel(l=acc @ t=5)
        //                          + Pblank(l=ac @ t=5))
        // else:
        //   Plabel(l=abc @ t=6) = (Plabel(l=abc @ t=5)
        //                          + P(l=ab @ t=5))
        float previous = (b->label == b->parent->label) ? b->parent->oldp.blank
                                                        : b->parent->oldp.total;
        b->newp.label =
            LogSumExp(b->newp.label,
                      beam_scorer_->GetStateExpansionScore(b->state, previous));
      }
      // Plabel(l=abc @ t=6) *= P(c @ 6)
      b->newp.label += raw_input(b->label) - norm_offset;
    }
    // Pblank(l=abc @ t=6) = P(l=abc @ t=5) * P(- @ 6)
    b->newp.blank = b->oldp.total + raw_input(blank_index_) - norm_offset;
    // P(l=abc @ t=6) = Plabel(l=abc @ t=6) + Pblank(l=abc @ t=6)
    b->newp.total = LogSumExp(b->newp.blank, b->newp.label);

    // Push the entry back to the top paths list.
    // Note, this will always fill leaves back up in sorted order.
    leaves_.push(b);
  }

  // we need to resort branches in descending oldp order.

  // branches is in descending oldp order because it was
  // originally in descending newp order and we copied newp to oldp.

  // Grow new leaves
  for (BeamEntry* b : *branches) {
    // A new leaf (represented by its BeamProbability) is a candidate
    // iff its total probability is nonzero and either the beam list
    // isn't full, or the lowest probability entry in the beam has a
    // lower probability than the leaf.
    auto is_candidate = [this](const BeamProbability& prob) {
      return (prob.total > kLogZero &&
              (leaves_.size() < beam_width_ ||
               prob.total > leaves_.peek_bottom()->newp.total));
    };

    if (!is_candidate(b->oldp)) {
      continue;
    }

    for (int ind = 0; ind < max_classes; ind++) {
      const int label = top_k ? top_k_indices[ind] : ind;
      const float logit = top_k ? top_k_logits[ind] : raw_input(ind);
      // Perform label selection: if input for this label looks very
      // unpromising, never evaluate it with a scorer.
      if (logit < label_selection_input_min) {
        continue;
      }
      BeamEntry& c = b->GetChild(label);
      if (!c.Active()) {
        //   Pblank(l=abcd @ t=6) = 0
        c.newp.blank = kLogZero;
        // If new child label is identical to beam label:
        //   Plabel(l=abcc @ t=6) = Pblank(l=abc @ t=5) * P(c @ 6)
        // Otherwise:
        //   Plabel(l=abcd @ t=6) = P(l=abc @ t=5) * P(d @ 6)
        float ext_score;
        beam_scorer_->ExpandState(b->state, b->label, &c.state, c.label);
        float previous = (c.label == b->label) ? b->oldp.blank : b->oldp.total;
        ext_score = beam_scorer_->GetStateExpansionScore(c.state, previous);
        c.newp.label = logit - norm_offset + ext_score;
        // P(l=abcd @ t=6) = Plabel(l=abcd @ t=6)
        c.newp.total = c.newp.label;

        if (is_candidate(c.newp)) {
          // Before adding the new node to the beam, check if the beam
          // is already at maximum width.
          if (leaves_.size() == beam_width_) {
            // Bottom is no longer in the beam search.  Reset
            // its probability; signal it's no longer in the beam search.
            BeamEntry* bottom = leaves_.peek_bottom();
            bottom->newp.Reset();
          }
          leaves_.push(&c);
        } else {
          // Deactivate child.
          c.oldp.Reset();
          c.newp.Reset();
        }
      }
    }
  }  // for (BeamEntry* b...
}

template <typename CTCBeamState, typename CTCBeamComparer>
void CTCBeamSearchNormLogDecoder<CTCBeamState, CTCBeamComparer>::Reset() {
  leaves_.Reset();

  // This beam root, and all of its children, will be in memory until
  // the next reset.
  beam_root_.reset(new BeamRoot(nullptr, -1));
  beam_root_->RootEntry()->newp.total = 0.0;  // ln(1)
  beam_root_->RootEntry()->newp.blank = 0.0;  // ln(1)

  // Add the root as the initial leaf.
  leaves_.push(beam_root_->RootEntry());

  // Call initialize state on the root object.
  beam_scorer_->InitializeState(&beam_root_->RootEntry()->state);
}

template <typename CTCBeamState, typename CTCBeamComparer>
Status CTCBeamSearchNormLogDecoder<CTCBeamState, CTCBeamComparer>::TopPaths(
    int n, std::vector<std::vector<int>>* paths, std::vector<float>* log_probs,
    bool merge_repeated) const {
  CHECK_NOTNULL(paths)->clear();
  CHECK_NOTNULL(log_probs)->clear();
  if (n > beam_width_) {
    return errors::InvalidArgument("requested more paths than the beam width.");
  }
  if (n > leaves_.size()) {
    return errors::InvalidArgument(
        "Less leaves in the beam search than requested.");
  }

  gtl::TopN<BeamEntry*, CTCBeamComparer> top_branches(n);

  // O(beam_width_ * log(n)), space complexity is O(n)
  for (auto it = leaves_.unsorted_begin(); it != leaves_.unsorted_end(); ++it) {
    top_branches.push(*it);
  }
  // O(n * log(n))
  std::unique_ptr<std::vector<BeamEntry*>> branches(top_branches.Extract());

  for (int i = 0; i < n; ++i) {
    BeamEntry* e((*branches)[i]);
    paths->push_back(e->LabelSeq(merge_repeated));
    log_probs->push_back(e->newp.total);
  }
  return Status::OK();
}


}
}















namespace tf = tensorflow;
using tf::shape_inference::DimensionHandle;
using tf::shape_inference::InferenceContext;
using tf::shape_inference::ShapeHandle;

REGISTER_OP("CTCBeamSearchDecoderWithLM")
    .Input("inputs: float")
    .Input("sequence_length: int32")
    .Attr("model_path: string")
    .Attr("trie_path: string")
    .Attr("alphabet_path: string")
    .Attr("alpha: float")
    .Attr("beta: float")
    .Attr("beam_width: int >= 1 = 100")
    .Attr("top_paths: int >= 1 = 1")
    .Attr("merge_repeated: bool = true")
    .Output("decoded_indices: top_paths * int64")
    .Output("decoded_values: top_paths * int64")
    .Output("decoded_shape: top_paths * int64")
    .Output("log_probability: float")
    .SetShapeFn([](InferenceContext *c) {
      ShapeHandle inputs;
      ShapeHandle sequence_length;

      TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 3, &inputs));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &sequence_length));

      // Get batch size from inputs and sequence_length.
      DimensionHandle batch_size;
      TF_RETURN_IF_ERROR(
          c->Merge(c->Dim(inputs, 1), c->Dim(sequence_length, 0), &batch_size));

      tf::int32 top_paths;
      TF_RETURN_IF_ERROR(c->GetAttr("top_paths", &top_paths));

      // Outputs.
      int out_idx = 0;
      for (int i = 0; i < top_paths; ++i) {  // decoded_indices
        c->set_output(out_idx++, c->Matrix(InferenceContext::kUnknownDim, 2));
      }
      for (int i = 0; i < top_paths; ++i) {  // decoded_values
        c->set_output(out_idx++, c->Vector(InferenceContext::kUnknownDim));
      }
      ShapeHandle shape_v = c->Vector(2);
      for (int i = 0; i < top_paths; ++i) {  // decoded_shape
        c->set_output(out_idx++, shape_v);
      }
      c->set_output(out_idx++, c->Matrix(batch_size, top_paths));
      return tf::Status::OK();
    })
    .Doc(R"doc(
Performs beam search decoding on the logits given in input.

A note about the attribute merge_repeated: For the beam search decoder,
this means that if consecutive entries in a beam are the same, only
the first of these is emitted.  That is, when the top path is "A B B B B",
"A B" is returned if merge_repeated = True but "A B B B B" is
returned if merge_repeated = False.

inputs: 3-D, shape: `(max_time x batch_size x num_classes)`, the logits.
sequence_length: A vector containing sequence lengths, size `(batch)`.
model_path: A string containing the path to the KenLM model file to use.
trie_path: A string containing the path to the trie file built from the vocabulary.
alphabet_path: A string containing the path to the alphabet file (see alphabet.h).
alpha: alpha hyperparameter of CTC decoder. LM weight.
beta: beta hyperparameter of CTC decoder. Word insertion weight.
beam_width: A scalar >= 0 (beam search beam width).
top_paths: A scalar >= 0, <= beam_width (controls output size).
merge_repeated: If true, merge repeated classes in output.
decoded_indices: A list (length: top_paths) of indices matrices.  Matrix j,
  size `(total_decoded_outputs[j] x 2)`, has indices of a
  `SparseTensor<int64, 2>`.  The rows store: [batch, time].
decoded_values: A list (length: top_paths) of values vectors.  Vector j,
  size `(length total_decoded_outputs[j])`, has the values of a
  `SparseTensor<int64, 2>`.  The vector stores the decoded classes for beam j.
decoded_shape: A list (length: top_paths) of shape vector.  Vector j,
  size `(2)`, stores the shape of the decoded `SparseTensor[j]`.
  Its values are: `[batch_size, max_decoded_length[j]]`.
log_probability: A matrix, shaped: `(batch_size x top_paths)`.  The
  sequence log-probabilities.
)doc");

class CTCDecodeHelper {
 public:
  CTCDecodeHelper() : top_paths_(1) {}

  inline int GetTopPaths() const { return top_paths_; }
  void SetTopPaths(int tp) { top_paths_ = tp; }

  tf::Status ValidateInputsGenerateOutputs(
      tf::OpKernelContext *ctx, const tf::Tensor **inputs, const tf::Tensor **seq_len,
      std::string *model_path, std::string *trie_path, std::string *alphabet_path,
      tf::Tensor **log_prob, tf::OpOutputList *decoded_indices,
      tf::OpOutputList *decoded_values, tf::OpOutputList *decoded_shape) const {
    tf::Status status = ctx->input("inputs", inputs);
    if (!status.ok()) return status;
    status = ctx->input("sequence_length", seq_len);
    if (!status.ok()) return status;

    const tf::TensorShape &inputs_shape = (*inputs)->shape();

    if (inputs_shape.dims() != 3) {
      return tf::errors::InvalidArgument("inputs is not a 3-Tensor");
    }

    const tf::int64 max_time = inputs_shape.dim_size(0);
    const tf::int64 batch_size = inputs_shape.dim_size(1);

    if (max_time == 0) {
      return tf::errors::InvalidArgument("max_time is 0");
    }
    if (!tf::TensorShapeUtils::IsVector((*seq_len)->shape())) {
      return tf::errors::InvalidArgument("sequence_length is not a vector");
    }

    if (!(batch_size == (*seq_len)->dim_size(0))) {
      return tf::errors::FailedPrecondition(
          "len(sequence_length) != batch_size.  ", "len(sequence_length):  ",
          (*seq_len)->dim_size(0), " batch_size: ", batch_size);
    }

    auto seq_len_t = (*seq_len)->vec<tf::int32>();

    for (int b = 0; b < batch_size; ++b) {
      if (!(seq_len_t(b) <= max_time)) {
        return tf::errors::FailedPrecondition("sequence_length(", b, ") <= ",
                                          max_time);
      }
    }

    tf::Status s = ctx->allocate_output(
        "log_probability", tf::TensorShape({batch_size, top_paths_}), log_prob);
    if (!s.ok()) return s;

    s = ctx->output_list("decoded_indices", decoded_indices);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_values", decoded_values);
    if (!s.ok()) return s;
    s = ctx->output_list("decoded_shape", decoded_shape);
    if (!s.ok()) return s;

    return tf::Status::OK();
  }

  // sequences[b][p][ix] stores decoded value "ix" of path "p" for batch "b".
  tf::Status StoreAllDecodedSequences(
      const std::vector<std::vector<std::vector<int>>> &sequences,
      tf::OpOutputList *decoded_indices, tf::OpOutputList *decoded_values,
      tf::OpOutputList *decoded_shape) const {
    // Calculate the total number of entries for each path
    const tf::int64 batch_size = sequences.size();
    std::vector<tf::int64> num_entries(top_paths_, 0);

    // Calculate num_entries per path
    for (const auto &batch_s : sequences) {
      CHECK_EQ(batch_s.size(), top_paths_);
      for (int p = 0; p < top_paths_; ++p) {
        num_entries[p] += batch_s[p].size();
      }
    }

    for (int p = 0; p < top_paths_; ++p) {
      tf::Tensor *p_indices = nullptr;
      tf::Tensor *p_values = nullptr;
      tf::Tensor *p_shape = nullptr;

      const tf::int64 p_num = num_entries[p];

      tf::Status s =
          decoded_indices->allocate(p, tf::TensorShape({p_num, 2}), &p_indices);
      if (!s.ok()) return s;
      s = decoded_values->allocate(p, tf::TensorShape({p_num}), &p_values);
      if (!s.ok()) return s;
      s = decoded_shape->allocate(p, tf::TensorShape({2}), &p_shape);
      if (!s.ok()) return s;

      auto indices_t = p_indices->matrix<tf::int64>();
      auto values_t = p_values->vec<tf::int64>();
      auto shape_t = p_shape->vec<tf::int64>();

      tf::int64 max_decoded = 0;
      tf::int64 offset = 0;

      for (tf::int64 b = 0; b < batch_size; ++b) {
        auto &p_batch = sequences[b][p];
        tf::int64 num_decoded = p_batch.size();
        max_decoded = std::max(max_decoded, num_decoded);
        std::copy_n(p_batch.begin(), num_decoded, &values_t(offset));
        for (tf::int64 t = 0; t < num_decoded; ++t, ++offset) {
          indices_t(offset, 0) = b;
          indices_t(offset, 1) = t;
        }
      }

      shape_t(0) = batch_size;
      shape_t(1) = max_decoded;
    }
    return tf::Status::OK();
  }

 private:
  int top_paths_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCDecodeHelper);
};

class CTCBeamSearchDecoderWithLMOp : public tf::OpKernel {
 public:
  explicit CTCBeamSearchDecoderWithLMOp(tf::OpKernelConstruction *ctx)
    : tf::OpKernel(ctx)
    , beam_scorer_(GetModelPath(ctx),
                   GetTriePath(ctx),
                   GetAlphabetPath(ctx),
                   GetAlpha(ctx),
                   GetBeta(ctx))
  {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("merge_repeated", &merge_repeated_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("beam_width", &beam_width_));
    int top_paths;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("top_paths", &top_paths));
    decode_helper_.SetTopPaths(top_paths);
  }

  void Compute(tf::OpKernelContext *ctx) override {
    const tf::Tensor *inputs;
    const tf::Tensor *seq_len;
    std::string model_path;
    std::string trie_path;
    std::string alphabet_path;
    tf::Tensor *log_prob = nullptr;
    tf::OpOutputList decoded_indices;
    tf::OpOutputList decoded_values;
    tf::OpOutputList decoded_shape;
    OP_REQUIRES_OK(ctx, decode_helper_.ValidateInputsGenerateOutputs(
                            ctx, &inputs, &seq_len, &model_path, &trie_path,
                            &alphabet_path, &log_prob, &decoded_indices,
                            &decoded_values, &decoded_shape));

    auto inputs_t = inputs->tensor<float, 3>();
    auto seq_len_t = seq_len->vec<tf::int32>();
    auto log_prob_t = log_prob->matrix<float>();

    const tf::TensorShape &inputs_shape = inputs->shape();

    const tf::int64 max_time = inputs_shape.dim_size(0);
    const tf::int64 batch_size = inputs_shape.dim_size(1);
    const tf::int64 num_classes_raw = inputs_shape.dim_size(2);
    OP_REQUIRES(
        ctx, tf::FastBoundsCheck(num_classes_raw, std::numeric_limits<int>::max()),
        tf::errors::InvalidArgument("num_classes cannot exceed max int"));
    const int num_classes = static_cast<const int>(num_classes_raw);

    log_prob_t.setZero();

    std::vector<tf::TTypes<float>::UnalignedConstMatrix> input_list_t;

    for (std::size_t t = 0; t < max_time; ++t) {
      input_list_t.emplace_back(inputs_t.data() + t * batch_size * num_classes,
                                batch_size, num_classes);
    }

    tf::ctc::CTCBeamSearchNormLogDecoder<WordLMBeamState> beam_search(num_classes, beam_width_,
                                            &beam_scorer_, 1 /* batch_size */,
                                            merge_repeated_);
    tf::Tensor input_chip(tf::DT_FLOAT, tf::TensorShape({num_classes}));
    auto input_chip_t = input_chip.flat<float>();

    std::vector<std::vector<std::vector<int>>> best_paths(batch_size);
    std::vector<float> log_probs;

    // Assumption: the blank index is num_classes - 1
    for (int b = 0; b < batch_size; ++b) {
      auto &best_paths_b = best_paths[b];
      best_paths_b.resize(decode_helper_.GetTopPaths());
      for (int t = 0; t < seq_len_t(b); ++t) {
        input_chip_t = input_list_t[t].chip(b, 0);
        auto input_bi =
            Eigen::Map<const Eigen::ArrayXf>(input_chip_t.data(), num_classes);
        beam_search.Step(input_bi);
      }

      typedef tf::ctc::ctc_beam_search::BeamEntry<WordLMBeamState> BeamEntry;
      std::unique_ptr<std::vector<BeamEntry*>> branches(beam_search.leaves_.Extract());
      beam_search.leaves_.Reset();
      for (int i = 0; i < branches->size(); ++i) {
        BeamEntry* entry = (*branches)[i];
        beam_scorer_.ExpandStateEnd(&entry->state);
        entry->newp.total +=
            beam_scorer_.GetStateEndExpansionScore(entry->state);
        beam_search.leaves_.push(entry);
      }
      
      OP_REQUIRES_OK(
          ctx, beam_search.TopPaths(decode_helper_.GetTopPaths(), &best_paths_b,
                                    &log_probs, merge_repeated_));

      beam_search.Reset();

      for (int bp = 0; bp < decode_helper_.GetTopPaths(); ++bp) {
        log_prob_t(b, bp) = log_probs[bp];
      }

    }

    OP_REQUIRES_OK(ctx, decode_helper_.StoreAllDecodedSequences(
                            best_paths, &decoded_indices, &decoded_values,
                            &decoded_shape));
  }

 private:
  CTCDecodeHelper decode_helper_;
  WordLMBeamScorer beam_scorer_;
  bool merge_repeated_;
  int beam_width_;
  TF_DISALLOW_COPY_AND_ASSIGN(CTCBeamSearchDecoderWithLMOp);

  std::string GetModelPath(tf::OpKernelConstruction *ctx) {
    std::string model_path;
    ctx->GetAttr("model_path", &model_path);
    return model_path;
  }

  std::string GetTriePath(tf::OpKernelConstruction *ctx) {
    std::string trie_path;
    ctx->GetAttr("trie_path", &trie_path);
    return trie_path;
  }

  std::string GetAlphabetPath(tf::OpKernelConstruction *ctx) {
    std::string alphabet_path;
    ctx->GetAttr("alphabet_path", &alphabet_path);
    return alphabet_path;
  }

  float GetAlpha(tf::OpKernelConstruction *ctx) {
    float alpha;
    ctx->GetAttr("alpha", &alpha);
    return alpha;
  }

  float GetBeta(tf::OpKernelConstruction *ctx) {
    float beta;
    ctx->GetAttr("beta", &beta);
    return beta;
  }

};

REGISTER_KERNEL_BUILDER(Name("CTCBeamSearchDecoderWithLM").Device(tf::DEVICE_CPU),
                        CTCBeamSearchDecoderWithLMOp);
