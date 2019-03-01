# DeepSpeech custom decoder

This folder contains a CTC beam search decoder that scores beams using a KenLM-based N-gram language model. 

## Installation

## Required Dependencies

Running inference might require some runtime dependencies to be already installed on your system. Those should be the same, whatever the bindings you are using:
* libsox2
* libstdc++6
* libgomp1
* libpthread


## Build Requirements

You'll need the following pre-requisites downloaded/installed:

* [TensorFlow source and requirements](https://www.tensorflow.org/install/install_sources)
* [libsox](https://sourceforge.net/projects/sox/)


## Preparation

Create a symbolic link in your TensorFlow checkout to `ctc_decoder_with_lm` directory. If your DeepSpeech and TensorFlow checkouts are side by side in the same directory, do:

```
cd tensorflow
ln -s ../OpenSeq2Seq/ctc_decoder_with_lm ./
```

## Building

## Step 1 : Build Tensorflow
You need to re-build TensorFlow.
Follow the [instructions](https://www.tensorflow.org/install/install_sources) on the TensorFlow site for your platform, up to the end of 'Configure the installation':

```
./configure
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package 
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow*.whl
sudo pip install --upgrade numpy
```

## Step 2: Build CTC beam search decoder:

```
bazel build -c opt --copt=-O3 --config=cuda //tensorflow:libtensorflow_cc.so //tensorflow:libtensorflow_framework.so //ctc_decoder_with_lm:libctc_decoder_with_kenlm.so //ctc_decoder_with_lm:generate_trie 
cp bazel-bin/ctc_decoder_with_lm/*.so OpenSeq2Seq/ctc_decoder_with_lm/
cp bazel-bin/ctc_decoder_with_lm/generate_trie OpenSeq2Seq/ctc_decoder_with_lm/
```


# Custom decoder with neural language model

## Step 1: Build Tensorflow 
- same as for DeepSpeech custom decoder above

## Step 1.1: Download and build LibPytorch from source
```
git clone https://github.com/pytorch/pytorch.git
cd pytorch
python -m tools.build_libtorch
```

## Step 1.2
In ```ctc_decoder_with_lm``` folder create symbolic link to
libtorch:

```
ln -s <path_to_pytorch_repo>/torch libtorch
```

## Step 1.5: New build file
```
mv BUILD.with-nm-lm BUILD
```

## Step 2: Build CTC beam search decoder:
- same as for DeepSpeech custom decoder above


## Step 3: Train and Trace Pytorch word-based LM
- your LM should be word based
- it should return log probability of a sentence `logP(w1,...,wN)`
- Your model should have absolute names like: `<path>/model.pt` and its vocabluary as `<path>/model.voc`

## Step 4: Run evaluation with new LM
In OS2S config, in the decoder section, set:
```
    "mode": 1,
    "neural_lm_path": "<path>/model"
```    


### Example on how to trace PyTorch LM:

```python
# This is a part which'll be traced.
class InferenceModel(nn.Module):
    def __init__(self, nn_model):
        super(InferenceModel, self).__init__()
        self._nn_model = nn_model
        self.softmax = torch.nn.Softmax(dim=2)


    def forward(self, input):
        input = input.long()
        # input is [time, batch]
        output, hidden = self._nn_model(input)
        # outputs are now [time, batch, vocab]
        prob_distr = self.softmax(output) # [time, batch, vocab]
        # select relevant probability scores
        probs = torch.gather(prob_distr, dim=2,
                             index=input.unsqueeze(-1)).squeeze(-1)
        # take logarithm
        return torch.sum(torch.log(probs))
```

...
```python
...
inf_model = InferenceModel(model.cpu())
dummy_input = torch.tensor([221, 222, 223, 26, 224]).unsqueeze(1)
inf_model.eval()
traced_script_module = torch.jit.trace(inf_model, dummy_input)
traced_script_module.save('<path>/model.pt')
...
```
