# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf

from collections import defaultdict

from open_seq2seq.utils.utils import get_base_config, check_logdir,\
                                     create_model, get_interactive_infer_results

from open_seq2seq.data.speech2text.speech_utils import get_speech_features_from_file, get_speech_features 

from ctc_decoders import Scorer, BeamDecoder



# Define the command line arguments that one would pass to run.py here
MODEL_PARAMS = ["--config_file=models/Jasper-Mini-for-Jetson/config_infer_stream.py",
                "--mode=interactive_infer",
                "--logdir=models/Jasper-Mini-for-Jetson/",
                "--batch_size_per_gpu=1",
                "--num_gpus=1",
                "--use_horovod=False",
                "--decoder_params/infer_logits_to_pickle=True",
                "--data_layer_params/pad_to=0"
]


def softmax(x):
    '''
    Naive softmax implementation for NumPy
    '''
    m = np.expand_dims(np.max(x, axis=-1), -1)
    e = np.exp(x - m)
    return e / np.expand_dims(e.sum(axis=-1), -1)


class FrameASR:
    
    def __init__(self, model_params=MODEL_PARAMS, scope_name='S2T', 
                 sr=16000, frame_len=0.2, frame_overlap=2.4, 
                 timestep_duration=0.02, 
                 ext_model_infer_func=None, merge=True,
                 beam_width=1, language_model=None, 
                 alpha=2.8, beta=1.0):
        '''
        Args:
          model_params: list of OpenSeq2Seq arguments (same as for run.py)
          scope_name: model's scope name
          sr: sample rate, Hz
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
            frame_overlap should be multiple of frame_len
          timestep_duration: time per step at model's output, seconds
          ext_model_infer_func: callback for external inference engine,
            if it is not None, then we don't build TF inference graph
          merge: whether to do merge in greedy decoder
          beam_width: beam width for beam search decoder if larger than 1
          language_model: path to LM (to use with beam search decoder)
          alpha: LM weight (trade-off between acoustic and LM scores)
          beta: word weight (added per every transcribed word in prediction)
        '''
        if ext_model_infer_func is None:
            # Build TF inference graph
            self.model_S2T, checkpoint_S2T = self._get_model(model_params, scope_name)

            # Create the session and load the checkpoints
            sess_config = tf.ConfigProto(allow_soft_placement=True)
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.InteractiveSession(config=sess_config)
            vars_S2T = {}
            for v in tf.get_collection(tf.GraphKeys.VARIABLES):
                if scope_name in v.name:
                    vars_S2T['/'.join(v.op.name.split('/')[1:])] = v
            saver_S2T = tf.train.Saver(vars_S2T)
            saver_S2T.restore(self.sess, checkpoint_S2T)
            self.params = self.model_S2T.params
        else:
            # No TF, load pre-, post-processing parameters from config,
            # use external inference engine
            _, base_config, _, _ = get_base_config(model_params)
            self.params = base_config

        self.ext_model_infer_func = ext_model_infer_func

        self.vocab = self._load_vocab(
            self.model_S2T.params['data_layer_params']['vocab_file']
        )
        self.sr = sr
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * sr)
        if self.n_frame_overlap % self.n_frame_len:
            raise ValueError(
                "'frame_overlap' should be multiple of 'frame_len'"
            )
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len, dtype=np.float32)
        self.merge = merge
        self._beam_decoder = None
        # greedy decoder's state (unmerged transcription)
        self.text = ''
        # forerunner greedy decoder's state (unmerged transcription)
        self.forerunner_text = ''

        self.offset = 5
        # self._calibrate_offset()
        if beam_width > 1:
          if language_model is None:
            self._beam_decoder = BeamDecoder(self.vocab, beam_width)
          else:
            self._scorer = Scorer(alpha, beta, language_model, self.vocab)
            self._beam_decoder = BeamDecoder(self.vocab, beam_width, ext_scorer=self._scorer)
        self.reset()


    def _get_audio(self, wav):
        """Parses audio from wav and returns array of audio features.
        Args:
            wav: numpy array containing wav
 
        Returns:
            tuple: source audio features as ``np.array``, length of source sequence,
            sample id.
        """
        source, audio_duration = get_speech_features(
            wav, 16000., self.params['data_layer_params']
        )

        return source, \
            np.int32([len(source)]), np.int32([0]), \
            np.float32([audio_duration])


    def _parse_audio_element(self, id_and_audio_filename):
        """Parses audio from file and returns array of audio features.
        Args:
            id_and_audio_filename: tuple of sample id and corresponding
            audio file name.
        Returns:
            tuple: source audio features as ``np.array``, length of source sequence,
            sample id.
        """
        idx, audio_filename = id_and_audio_filename
        source, audio_duration = get_speech_features_from_file(
            audio_filename,
            params=self.params
        )
        return source, \
            np.int32([len(source)]), np.int32([idx]), \
            np.float32([audio_duration])


    def _preprocess_audio(self, model_in):
        audio_arr = []
        audio_length_arr = []

        for line in model_in:
          if isinstance(line, str):
            features, features_length, _, _ = self._parse_audio_element([0, line])
          elif isinstance(line, np.ndarray):
            features, features_length, _, _ = self._get_audio(line)
          else:
            raise ValueError(
                "Speech2Text's interactive inference mode only supports string or",
                "numpy array as input. Got {}". format(type(line))
            )
          audio_arr.append(features)
          audio_length_arr.append(features_length)
        max_len = np.max(audio_length_arr)
        pad_to = self.params.get("pad_to", 8)
        if pad_to > 0 and self.params.get('backend') == 'librosa':
          max_len += (pad_to - max_len % pad_to) % pad_to
        for idx in range(len(audio_arr)):
          audio_arr[idx] = np.pad(
              audio_arr[idx], ((0, max_len-len(audio_arr[idx])), (0, 0)),
              "constant", constant_values=0.
          )

        audio_features = np.reshape(
            audio_arr,
            [self.params['batch_size_per_gpu'],
             -1,
             self.params['data_layer_params']['num_audio_features']]
        )
        features_length = np.reshape(audio_length_arr, [self.params['batch_size_per_gpu']])
        return [audio_features, features_length]


    def _decode(self, frame, offset=0, merge=False):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame

        audio_features, features_length = self._preprocess_audio([self.buffer])
        if self.ext_model_infer_func is None:
            logits = get_interactive_infer_results(
                self.model_S2T, self.sess, 
                model_in={'audio_features': audio_features,
                          'features_length': features_length})[0][0]
        else:
            # TODO: check ext_model_infer_func parameters and return value
            logits = self.ext_model_infer_func(audio_features, features_length)

        if self._beam_decoder is None:
          decoded_forerunner = self._greedy_decoder(
              logits[self.n_timesteps_overlap:], 
              self.vocab
          )
          decoded = decoded_forerunner[:-self.n_timesteps_overlap-offset]

          forerunner_idx = max(0, len(self.forerunner_text) - \
              (self.n_timesteps_overlap + offset))
          self.forerunner_text = self.forerunner_text[:forerunner_idx] + \
              decoded_forerunner
          self.text += decoded
          if merge:
            decoded = self.greedy_merge(self.text)
            decoded_forerunner = self.greedy_merge(self.forerunner_text)
        else:
          decoded = self._beam_decoder.decode(softmax(
              logits[self.n_timesteps_overlap:-self.n_timesteps_overlap-offset]
          ))[0][-1]

        return [decoded, decoded_forerunner]

    
    def transcribe(self, frame=None):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        return self._decode(frame, self.offset, self.merge)
    
    
    def _calibrate_offset(self, wav_file, max_offset=10, n_calib_inter=100):
        '''
        Calibrate offset for frame-by-frame decoding
        '''
        sr, signal = wave.read(wav_file)
        
        # warmup
        n_warmup = 1 + int(np.ceil(2.0 * self.frame_overlap / self.frame_len))
        for i in range(n_warmup):
            decoded, _ = self._decode(signal[self.n_frame_len*i:self.n_frame_len*(i+1)], offset=0)
        
        i = n_warmup
        
        offsets = defaultdict(lambda: 0)
        while i < n_warmup + n_calib_inter and (i+1)*self.n_frame_len < signal.shape[0]:
            decoded_prev = decoded
            decoded, _ = self._decode(signal[self.n_frame_len*i:self.n_frame_len*(i+1)], offset=0)
            for offset in range(max_offset, 0, -1):
                if decoded[:offset] == decoded_prev[-offset:] and decoded[:offset] != ''.join(['_']*offset):
                    offsets[offset] += 1
                    break
            i += 1
        self.offset = max(offsets, key=offsets.get)
       
        
    def reset(self):
        '''
        Reset frame_history and decoder's state
        '''
        self.buffer=np.zeros(shape=self.buffer.shape, dtype=np.float32)
        if self._beam_decoder is not None:
            self._beam_decoder.reset()
        self.prev_char = ''
        self.text = ''
        self.forerunner_text = ''
        
    @staticmethod
    def _get_model(args, scope):
        '''
        A simpler version of what run.py does. It returns the created model and its saved checkpoint
        '''
        with tf.variable_scope(scope):
            args, base_config, base_model, config_module = get_base_config(args)
            checkpoint = check_logdir(args, base_config)
            model = create_model(args, base_config, config_module, base_model, None)
        return model, checkpoint

    @staticmethod
    def _load_vocab(vocab_file):
        vocab = []
        with open(vocab_file, 'r') as f:
            for line in f:
                vocab.append(line[0])
        vocab.append('_')
        return vocab

    @staticmethod
    def _greedy_decoder(logits, vocab):
        s = ''
        for i in range(logits.shape[0]):
            s += vocab[np.argmax(logits[i])]
        return s

    def greedy_merge(self, s, prev_char=''):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != prev_char:
                prev_char = s[i]
                if prev_char != '_':
                    s_merged += prev_char
        return s_merged

