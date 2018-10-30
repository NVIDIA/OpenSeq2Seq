# Replace the first box of Interactive_Infer_example.ipynb with this

import IPython
import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_model, get_interactive_infer_results
from open_seq2seq.models.text2speech_wavenet import save_audio

args_T2S = ["--config_file=Infer_T2S_Wave/config.py",
        "--mode=interactive_infer",
        "--logdir=Infer_T2S_Wave/",
        "--batch_size_per_gpu=1",
]

# A simpler version of what run.py does. It returns the created model and its 
# saved checkpoint
def get_model(args, scope):
    with tf.variable_scope(scope):
        args, base_config, base_model, config_module = get_base_config(args)
        checkpoint = check_logdir(args, base_config)
        model = create_model(args, base_config, config_module, base_model, None)
    return model, checkpoint

model_T2S, checkpoint_T2S = get_model(args_T2S, "T2S")

# Create the session and load the checkpoints
sess_config = tf.ConfigProto(allow_soft_placement=True)
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

vars_T2S = {}
for v in tf.get_collection(tf.GraphKeys.VARIABLES):
    if "T2S" in v.name:
        vars_T2S["/".join(v.op.name.split("/")[1:])] = v

saver_T2S = tf.train.Saver(vars_T2S)
saver_T2S.restore(sess, checkpoint_T2S)

# Define the inference function
n_fft = model_T2S.get_data_layer().n_fft
sampling_rate = model_T2S.get_data_layer().sampling_rate
def infer(line):
    """
    Infers one value at a time using a sliding window with width equal to the
    receptive field. 
    """

    print("Input File")
    print(line) 
    
    GET_SPEC_FROM_WAV = False
    max_steps = 200000
    receptive_field = 6139 # 3x10
    
    source = np.zeros([1, receptive_field])
    src_length = np.full([1], receptive_field)
    audio = []
    spec_offset = 0
    
    if GET_SPEC_FROM_WAV: # get spectrogram from .wav file
        file_name = str.encode(line)
        spec, spec_length = model_T2S.get_data_layer(). \
            _parse_spectrogram_element(file_name)
    
    else: # get spectrogram from .npy file
        spec = np.load(line + ".npy").T
        spec = np.repeat(spec, 256, axis=0)
        spec_length = spec.shape[0]

    spec = np.expand_dims(spec, axis=0)
    spec_length = np.reshape(spec_length, [1])
    
    while(spec_offset < max_steps):
        output = get_interactive_infer_results(
            model_T2S, sess, 
            model_in=(source, src_length, spec, spec_length, spec_offset)
        )
        
        predicted = output[-1][0]
        audio.append(predicted)
        
        source[0][0] = predicted
        source[0] = np.roll(source[0], -1)
        
        if spec_offset % 1000 == 0:
            print("Saving audio for step {}".format(spec_offset))
            wav = save_audio(
                np.array(audio), "result", 0, 
                sampling_rate=sampling_rate, mode="infer"
            )
            
        spec_offset += 1
