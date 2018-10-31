%matplotlib inline
# Replace the first box of Interactive_Infer_example.ipynb with this

import IPython
import librosa

import numpy as np
import scipy.io.wavfile as wave
import tensorflow as tf
import matplotlib.pyplot as plt

from open_seq2seq.utils.utils import deco_print, get_base_config, check_logdir,\
                                     create_logdir, create_model, get_interactive_infer_results
from open_seq2seq.models.text2speech import save_audio

args_T2S = ["--config_file=Infer_T2S/config.py",
        "--mode=interactive_infer",
        "--logdir=Infer_T2S/",
        "--batch_size_per_gpu=1",
]

# A simpler version of what run.py does. It returns the created model and its saved checkpoint
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

# line = "I was trained using Nvidia's Open Sequence to Sequence framework."

# Define the inference function
n_fft = model_T2S.get_data_layer().n_fft
sampling_rate = model_T2S.get_data_layer().sampling_rate
def infer(line):
    print("Input English")
    print(line)
    
    # Generate speech
    results = get_interactive_infer_results(model_T2S, sess, model_in=[line])
    audio_length = results[1][4][0]

    if model_T2S.get_data_layer()._both:
        prediction = results[1][5][0]

    else:
        prediction = results[1][1][0]

    prediction = prediction[:audio_length-1,:]
    mag_prediction = model_T2S.get_data_layer().get_magnitude_spec(prediction)

    mag_prediction_squared = np.clip(mag_prediction, a_min=0, a_max=255)
    mag_prediction_squared = mag_prediction_squared**1.5
    mag_prediction_squared = np.square(mag_prediction_squared)

    mel_basis = librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, htk=True, norm=None)
    mel = np.dot(mel_basis, mag_prediction_squared.T)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    np.save("spec2", mel)

    plt.imshow(mel)
    plt.gca().invert_yaxis()
    plt.show()

    wav = save_audio(mag_prediction, "unused", "unused", sampling_rate=sampling_rate, save_format="np.array", n_fft=n_fft)
    audio = IPython.display.Audio(wav, rate=sampling_rate)
    print("Generated Audio")
    IPython.display.display(audio)
