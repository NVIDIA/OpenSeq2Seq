import abc
import six
import sys
import os
import random
import copy
import numpy as np

@six.add_metaclass(abc.ABCMeta)
class DataLayer:
	UNK_ID = 0  # out-of-vocabulary tokens will map there
	S_ID = 1  # special start of sentence token
	EOS_ID = 2  # special end of sentence token
	PAD_ID = 3  # special padding token
	OUT_OF_BUCKET = 1234567890
	END_OF_CHOICE = -100
	"""Abstract class that specifies data access operations
	"""
	@abc.abstractmethod
	def __init__(self, params):
		"""Initialize data layer
		:param params: Python dictionary with options,
		specifying mini-batch shapes, padding, etc.
		"""
		self._params = params

	@abc.abstractmethod
	def iterate_one_epoch(self):
		"""
		Goes through the data one time.
		:return: yields rectangular 2D numpy array with mini-batch data
		"""
		pass

	@property
	def params(self):
		return self._params

class ParallelDataInRamInputLayer(DataLayer):
	"""Parallel data layer class. It should be provided with:
	a) vocabulary file
	b) folder with tokenized source files
	c) folder with tokenized target files
	This class performs:
		1) Loading of data, mapping tokens to their ids. 
		2) Inserting special tokens, if needed. 
		3) Padding. 
		4) Bucketing. 
		5) Mini-batch construction. 
	All parameters for above actions should come through "params" dictionary
	passed to constructions. 
	This class loads all data and serves it from RAM. 
	"""
	UNK_ID = 0 # out-of-vocabulary tokens will be map there
	S_ID = 1 # special start of sentence token
	EOS_ID = 2 # special end of sentence token
	PAD_ID = 3 # special padding token
	OUT_OF_BUCKET = 1234567890
	END_OF_CHOICE = -100

	def __init__(self, params):
		super().__init__(params)

		self.batch_size = self.params['batch_size'] * self.params['num_gpus'] if 'num_gpus' in self.params else self.params['batch_size']

		self.source_file = self.params['source_file']
		self.target_file = self.params['target_file']
		self.target_candidate_file = self.params['target_candidate_file'] if 'target_candidate_file' in self.params else None

		self.bucket_ctx = self.params['bucket_ctx']
		self.bucket_ques = self.params['bucket_ques']
		self.bucket_ans = self.params['bucket_ans']

		self._bucket_order = [] #used in inference
		self._shuffle = ('shuffle' in self._params and self._params['shuffle'])
		self._max_word_len = self._params['max_word_len'] if 'max_word_len' in self._params else 16
		self._mute_warning = self._params['mute_warning'] if 'mute_warning' in self._params else False

		self.custom_delimiter = self.params['delimiter']

		self.load_pretrained_embedding(self.params['vocab_embedding_file'], self.params['vocab_embedding_dim'])
		self.create_idx_seq_associations()
		self.create_char_idx_seq_associations()
		self.load_corpus()
		self.bucketize()

	def load_pretrained_embedding(self, path, dim):
		embedding = []
		vocab = {}
		idx = 4
		with open(path, 'r') as f:
			for line in f:
				line_split = line.rstrip('\n').split()
				vocab[line_split[0]] = idx
				idx += 1
				word_vec = [float(num) for num in line_split[1:]]
				embedding.append(word_vec)
		self.seq2idx = vocab
		self.embedding = np.array(embedding, dtype='float32')

	def load_corpus(self):
		self.context_corpus, self.question_corpus, self.context_corpus_char, self.question_corpus_char = self.load_file_with_special_symbols(self.source_file, self.seq2idx, source = True)
		self.answer_corpus, self.answer_position, self.answer_candidate = self.load_file_with_special_symbols(self.target_file, self.seq2idx, source = False, path_target_candidate = self.target_candidate_file)
		context_corpus_temp = []
		context_corpus_char_temp = []
		question_corpus_temp = []
		question_corpus_char_temp = []
		answer_corpus_temp = []
		answer_position_temp = []
		answer_candidate_temp = []
		for ctx, ctx_char, ques, ques_char, ans, pos, cand in zip(self.context_corpus, self.context_corpus_char, self.question_corpus, self.question_corpus_char, self.answer_corpus, self.answer_position, self.answer_candidate):
			if pos[0] != -1:
				context_corpus_temp.append(ctx)
				context_corpus_char_temp.append(ctx_char)
				question_corpus_temp.append(ques)
				question_corpus_char_temp.append(ques_char)
				answer_corpus_temp.append(ans)
				answer_position_temp.append(pos)
				answer_candidate_temp.append(cand)
		self.context_corpus, self.context_corpus_char, self.question_corpus, self.question_corpus_char, self.answer_corpus, self.answer_position, self.answer_candidate = context_corpus_temp, context_corpus_char_temp, question_corpus_temp, question_corpus_char_temp, answer_corpus_temp, answer_position_temp, answer_candidate_temp

	def load_file_with_special_symbols(self, path, vocab, source = True, path_target_candidate = None):
		if not source:
			answers = []
			positions = []
			ans_candidates = []
			with open(path, 'r') as f:
				for line in f:
					text_list = line.rstrip('\n').split('\t')
					answers.append([ParallelDataInRamInputLayer.S_ID] + list(
						map(lambda word: vocab[word] if word in vocab else ParallelDataInRamInputLayer.UNK_ID, text_list[0].split())) +
						[ParallelDataInRamInputLayer.EOS_ID])
					if len(text_list) > 1:
						positions.append((int(text_list[1]), int(text_list[2])))
					else:
						positions.append((-1, -1))
			if path_target_candidate:
				print('Loading Candidate Answers From: ' + path_target_candidate)
				with open(path_target_candidate, 'r') as f:
					for line in f:
						temp = line.rstrip('\n').split('\t')
						ans_candidates.append([list(map(lambda word: vocab[word] if word in vocab else ParallelDataInRamInputLayer.UNK_ID, text.split())) for text in temp])
			else:
				ans_candidates = [[answer[1:-1]] for answer in answers]
			return (answers, positions, ans_candidates)
		else:
			contexts = []
			questions = []
			contexts_char = []
			questions_char = []
			with open(path, 'r') as f:
				for line in f:
					temp = line.rstrip('\n').lstrip('<P>').split('<Q>')
					contexts.append([ParallelDataInRamInputLayer.S_ID] + list(
						map(lambda word: vocab[word] if word in vocab else ParallelDataInRamInputLayer.UNK_ID, temp[0].split())) +
						[ParallelDataInRamInputLayer.EOS_ID])
					questions.append([ParallelDataInRamInputLayer.S_ID] + list(
						map(lambda word: vocab[word] if word in vocab else ParallelDataInRamInputLayer.UNK_ID, temp[1].split())) +
						[ParallelDataInRamInputLayer.EOS_ID])
					contexts_char.append([[1]] + list(
						map(lambda word: [self.char2idx[ch] if ch in self.char2idx else self.char2idx['<unk>'] for ch in word], temp[0].split())) + 
						[[2]])
					questions_char.append([[1]] + list(
						map(lambda word: [self.char2idx[ch] if ch in self.char2idx else self.char2idx['<unk>'] for ch in word], temp[1].split())) + 
						[[2]])
			return (contexts, questions, contexts_char, questions_char)

	def create_idx_seq_associations(self):
		self.seq2idx['<UNK>'] = ParallelDataInRamInputLayer.UNK_ID
		self.seq2idx['<S>'] = ParallelDataInRamInputLayer.S_ID
		self.seq2idx['</S>'] = ParallelDataInRamInputLayer.EOS_ID
		self.seq2idx['<PAD>'] = ParallelDataInRamInputLayer.PAD_ID
		self.idx2seq = {id: w for w, id in self.seq2idx.items()}

	def create_char_idx_seq_associations(self):
		self.char2idx = {'<unk>':0, '<S>': 1, '</S>': 2, '<pad>': 3}
		for i in range(33, 127):
			self.char2idx[chr(i)] = i - 29
		self.idx2char = {id: ch for ch, id in self.char2idx.items()}

	def determine_bucket(self, input_size, bucket_sizes):
		if len(bucket_sizes) <= 0:
			raise ValueError("No buckets specified")
		curr_bucket = 0
		while curr_bucket<len(bucket_sizes) and input_size > bucket_sizes[curr_bucket]:
			curr_bucket += 1
		if curr_bucket >= len(bucket_sizes):
			return ParallelDataInRamInputLayer.OUT_OF_BUCKET
		else:
			return curr_bucket

	def bucketize(self):
		self._bucket_id_to_context_example = {}
		self._bucket_id_to_question_example = {}
		self._bucket_id_to_answer_example = {}
		self._bucket_id_to_answer_candidate_example = {}

		self._bucket_id_to_position_example = {}

		self._bucket_id_to_context_char_example = {}
		self._bucket_id_to_question_char_example = {}

		for ctx, ctx_char, ques, ques_char, ans, pos, ans_cand in zip(self.context_corpus, self.context_corpus_char, self.question_corpus, self.question_corpus_char, self.answer_corpus, self.answer_position, self.answer_candidate):
			bucket_id = max(self.determine_bucket(len(ctx), self.bucket_ctx),
							self.determine_bucket(len(ques), self.bucket_ques),
							self.determine_bucket(len(ans), self.bucket_ans))

			if bucket_id == ParallelDataInRamInputLayer.OUT_OF_BUCKET:
				if not self._mute_warning:
					print("WARNING: skipped pair with sizes of (%d, %d, %d)" % (len(ctx), len(ques), len(ans)))
				continue
			if not bucket_id in self._bucket_id_to_context_example:
				self._bucket_id_to_context_example[bucket_id] = []
			self._bucket_id_to_context_example[bucket_id].append(ctx)

			if not bucket_id in self._bucket_id_to_context_char_example:
				self._bucket_id_to_context_char_example[bucket_id] = []
			self._bucket_id_to_context_char_example[bucket_id].append(ctx_char)

			if not bucket_id in self._bucket_id_to_question_example:
				self._bucket_id_to_question_example[bucket_id] = []
			self._bucket_id_to_question_example[bucket_id].append(ques)

			if not bucket_id in self._bucket_id_to_question_char_example:
				self._bucket_id_to_question_char_example[bucket_id] = []
			self._bucket_id_to_question_char_example[bucket_id].append(ques_char)

			if not bucket_id in self._bucket_id_to_answer_example:
				self._bucket_id_to_answer_example[bucket_id] = []
			self._bucket_id_to_answer_example[bucket_id].append(ans)

			if not bucket_id in self._bucket_id_to_position_example:
				self._bucket_id_to_position_example[bucket_id] = []
			self._bucket_id_to_position_example[bucket_id].append(pos)

			if not bucket_id in self._bucket_id_to_answer_candidate_example:
				self._bucket_id_to_answer_candidate_example[bucket_id] = []
			self._bucket_id_to_answer_candidate_example[bucket_id].append(ans_cand)

			if not self._shuffle:
				self._bucket_order.append(bucket_id)

		self._bucket_sizes = {}
		for bucket_id in self._bucket_id_to_context_example.keys():
			self._bucket_sizes[bucket_id] = len(self._bucket_id_to_context_example[bucket_id])
			if self._shuffle:
				temp = list(zip(self._bucket_id_to_context_example[bucket_id],
								self._bucket_id_to_question_example[bucket_id],
								self._bucket_id_to_answer_example[bucket_id],
								self._bucket_id_to_position_example[bucket_id],
								self._bucket_id_to_answer_candidate_example[bucket_id],
								self._bucket_id_to_context_char_example[bucket_id],
								self._bucket_id_to_question_char_example[bucket_id]))
				random.shuffle(temp)
				a, b, c, d, e, a_char, b_char = zip(*temp)
				self._bucket_id_to_position_example[bucket_id] = np.asarray(d)

				self._bucket_id_to_context_char_example[bucket_id] = np.asarray(a_char)
				self._bucket_id_to_question_char_example[bucket_id] = np.asarray(b_char)

				self._bucket_id_to_context_example[bucket_id] = np.asarray(a)
				self._bucket_id_to_question_example[bucket_id] = np.asarray(b)
				self._bucket_id_to_answer_example[bucket_id] = np.asarray(c)
				self._bucket_id_to_answer_candidate_example[bucket_id] = np.asarray(e)
			else:
				self._bucket_id_to_context_example[bucket_id] = np.asarray(self._bucket_id_to_context_example[bucket_id])
				self._bucket_id_to_question_example[bucket_id] = np.asarray(self._bucket_id_to_question_example[bucket_id])
				self._bucket_id_to_answer_example[bucket_id] = np.asarray(self._bucket_id_to_answer_example[bucket_id])
				self._bucket_id_to_answer_candidate_example[bucket_id] = np.asarray(self._bucket_id_to_answer_candidate_example[bucket_id])
				self._bucket_id_to_position_example[bucket_id] = np.asarray(self._bucket_id_to_position_example[bucket_id])
				self._bucket_id_to_context_char_example[bucket_id] = np.asarray(self._bucket_id_to_context_char_example[bucket_id])
				self._bucket_id_to_question_char_example[bucket_id] = np.asarray(self._bucket_id_to_question_char_example[bucket_id])

	def _pad_to_bucket_size(self, inseq, bucket_size):
		if len(inseq) == bucket_size:
			return inseq
		else:
			return inseq + [ParallelDataInRamInputLayer.PAD_ID] * (bucket_size - len(inseq))

	def _pad_to_maximum_length(self, sequence, bucket_size, max_len = 5):
		outseq = []
		for inseq in sequence:
			if len(inseq) >= max_len:
				outseq.append((inseq[:max_len], max_len))
			else:
				outseq.append((inseq + [3] * (max_len - len(inseq)), len(inseq)))
		if len(outseq) < bucket_size:
			outseq = outseq + [([3] * max_len, 0)] * (bucket_size - len(outseq))
		return outseq

	def iterate_one_epoch(self):
		start_inds = {}
		choices = copy.deepcopy(self._bucket_sizes)
		for bucket_id in choices.keys():
			start_inds[bucket_id] = 0

		if self._shuffle:
			bucket_id = self.weighted_choice(choices)
		else:
			ordering = list(reversed(self._bucket_order))
			if len(ordering) >= self.batch_size:
				for _ in range(self.batch_size):
					bucket_id = ordering.pop()
			else:
				bucket_id = ordering.pop()
				ordering = []

		while bucket_id != self.END_OF_CHOICE:
			end_ind = min(start_inds[bucket_id] + self.batch_size, self._bucket_id_to_context_example[bucket_id].shape[0])
			x = self._bucket_id_to_context_example[bucket_id][start_inds[bucket_id]:end_ind]
			y = self._bucket_id_to_question_example[bucket_id][start_inds[bucket_id]:end_ind]
			len_x = np.asarray(list(map(lambda row: len(row), x)))
			len_y = np.asarray(list(map(lambda row: len(row), y)))
			x = np.vstack(map(lambda row: np.asarray(self._pad_to_bucket_size(list(row), self.bucket_ctx[bucket_id])), x))
			y = np.vstack(map(lambda row: np.asarray(self._pad_to_bucket_size(list(row), self.bucket_ques[bucket_id])), y))

			x_char = self._bucket_id_to_context_char_example[bucket_id][start_inds[bucket_id]:end_ind]
			y_char = self._bucket_id_to_question_char_example[bucket_id][start_inds[bucket_id]:end_ind]

			x_char_temp = list(map(lambda row: self._pad_to_maximum_length(row, self.bucket_ctx[bucket_id], self._max_word_len), x_char))
			x_char_temp = list(zip(*row) for row in x_char_temp)
			x_temp1, x_temp2 = zip(*x_char_temp)
			x_char = np.asarray(x_temp1)
			len_x_char = np.asarray(x_temp2)

			y_char_temp = list(map(lambda row: self._pad_to_maximum_length(row, self.bucket_ques[bucket_id], self._max_word_len), y_char))
			y_char_temp = list(zip(*row) for row in y_char_temp)
			y_temp1, y_temp2 = zip(*y_char_temp)
			y_char = np.asarray(y_temp1)
			len_y_char = np.asarray(y_temp2)

			z = self._bucket_id_to_answer_example[bucket_id][start_inds[bucket_id]:end_ind]
			len_z = np.asarray(list(map(lambda row: len(row), z)))
			z = np.vstack(map(lambda row: np.asarray(self._pad_to_bucket_size(list(row), self.bucket_ans[bucket_id])), z))
			z_pos = self._bucket_id_to_position_example[bucket_id][start_inds[bucket_id]:end_ind]

			z_cand = self._bucket_id_to_answer_candidate_example[bucket_id][start_inds[bucket_id]:end_ind]

			yield_examples = end_ind - start_inds[bucket_id]
			start_inds[bucket_id] += yield_examples

			bucket_id_to_yield = bucket_id
			choices[bucket_id] -= yield_examples

			if self._shuffle:
				bucket_id = self.weighted_choice(choices)
				if yield_examples < self.batch_size:
					continue
			elif len(ordering) >= self.batch_size:
				for _ in range(self.batch_size):
					bucket_id = ordering.pop()
			else:
				bucket_id = self.END_OF_CHOICE

			yield x, x_char, y, y_char, z, z_pos, bucket_id_to_yield, len_x, len_x_char, len_y, len_y_char, len_z, z_cand

	def pretty_print_array(self, row, vocab, ignore_special=False, delim=' '):
		if ignore_special:
			f_row = []
			for i in range(0, len(row)):
				char_id = row[i]
				if char_id==self.EOS_ID:
					break
				if char_id!=self.PAD_ID and char_id!=self.S_ID:
					f_row += [char_id]
			return delim.join(map(lambda x: vocab[x], [r for r in f_row if r > 0]))
		else:
			return delim.join(map(lambda x: vocab[x], [r for r in row if r > 0]))

	def weighted_choice(self, choices):
		total_weights = sum(w for c, w in choices.items())
		if total_weights <= 0:
			return self.END_OF_CHOICE
		r = random.uniform(0, total_weights)
		mx = 0
		for i, w in choices.items():
			if mx + w >= r:
				return i
			mx += w
		raise AssertionError("weighted choice got to the wrong place")