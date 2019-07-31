# Copyright (c) 2019 NVIDIA Corporation
from __future__ import absolute_import, division, print_function

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf

from collections import defaultdict

from open_seq2seq.utils.utils import get_base_config, check_logdir,\
                                     create_model, get_interactive_infer_results

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


class FrameASR:
    
    def __init__(self, model_params=MODEL_PARAMS, scope_name='S2T', 
                 sr=16000, frame_len=2, frame_overlap=2.5, 
                 timestep_duration=0.02):
        '''
        Args:
          model_params: list of OpenSeq2Seq arguments (same as for run.py)
          scope_name: model's scope name
          sr: sample rate, Hz
          frame_len: frame's duration, seconds
          frame_overlap: duration of overlaps before and after current frame, seconds
          timestep_duration: time per step at model's output, seconds
        '''
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
        
        self.vocab = self._load_vocab(
            self.model_S2T.params['data_layer_params']['vocab_file']
        )
        self.sr = sr
        self.frame_len = frame_len
        self.n_frame_len = int(frame_len * sr)
        self.frame_overlap = frame_overlap
        self.n_frame_overlap = int(frame_overlap * sr)
        self.n_timesteps_overlap = int(frame_overlap / timestep_duration) - 2
        self.buffer = np.zeros(shape=2*self.n_frame_overlap + self.n_frame_len, dtype=np.float32)
        # self._calibrate_offset()
        self.offser = 5
        self.reset()
        
        
    def _decode(self, frame, offset=0):
        assert len(frame)==self.n_frame_len
        self.buffer[:-self.n_frame_len] = self.buffer[self.n_frame_len:]
        self.buffer[-self.n_frame_len:] = frame
        logits = get_interactive_infer_results(
            self.model_S2T, self.sess, model_in=[self.buffer])[0][0]
        decoded = self._greedy_decoder(
            logits[self.n_timesteps_overlap:-self.n_timesteps_overlap], 
            self.vocab
        )
        return decoded[:len(decoded)-offset]
    
    def transcribe(self, frame=None, merge=True):
        if frame is None:
            frame = np.zeros(shape=self.n_frame_len, dtype=np.float32)
        if len(frame) < self.n_frame_len:
            frame = np.pad(frame, [0, self.n_frame_len - len(frame)], 'constant')
        unmerged = self._decode(frame, self.offset)
        if not merge:
            return unmerged
        return self.greedy_merge(unmerged)
    
    
    def _calibrate_offset(self, wav_file, max_offset=10, n_calib_inter=10):
        '''
        Calibrate offset for frame-by-frame decoding
        '''
        sr, signal = wave.read(wav_file)
        
        # warmup
        n_warmup = 1 + int(np.ceil(2.0 * self.frame_overlap / self.frame_len))
        for i in range(n_warmup):
            decoded = self._decode(signal[self.n_frame_len*i:self.n_frame_len*(i+1)], offset=0)
        
        i = n_warmup
        
        offsets = defaultdict(lambda: 0)
        while i < n_warmup + n_calib_inter and (i+1)*self.n_frame_len < signal.shape[0]:
            decoded_prev = decoded
            decoded = self._decode(signal[self.n_frame_len*i:self.n_frame_len*(i+1)], offset=0)
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
        self.prev_char = ''


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

    def greedy_merge(self, s):
        s_merged = ''
        
        for i in range(len(s)):
            if s[i] != self.prev_char:
                self.prev_char = s[i]
                if self.prev_char != '_':
                    s_merged += self.prev_char
        return s_merged

