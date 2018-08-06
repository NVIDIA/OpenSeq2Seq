.. _speech_synthesis:

Speech Synthesis
================

The table below contains description and results of
text-to-speech models available in OpenSeq2Seq.
Currently, we have a Tacotron2-based model.

.. list-table::
   :widths: 1 1 1 1 1
   :header-rows: 1

   * - Config file
     - Samples
     - Training setup and additional comments
     - Short description of the model
     - Checkpoint
   * - `tacotron_LJ_float.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_float.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 1 gpu for 100,000 steps with ADAM.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.
   * - `tacotron_LJ_float_8gpu.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_float_8gpu.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 8 gpus for 30,000 steps with ADAM and larc.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.
   * - `tacotron_LJ_mixed.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_LJ_mixed.py>`_
     - Coming soon.
     - Learns magnitude spectrograms. Trained on 1 gpu for 100,000 steps with ADAM and larc.
     - Model tries to match the model description in https://arxiv.org/abs/1712.05884.
       The only difference is that the stop token projection layer is placed after
       the spectrogram projection layer.
     - Coming soon.


Tacotron 2 model description: https://arxiv.org/abs/1712.05884.
