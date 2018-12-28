.. _jasper:

Jasper
=======

Model
~~~~~~

Jasper (Just Another Speech Recognizer) is a deep time delay neural network (TDNN) comprising of blocks of 1D-convolutional layers. Jasper is a family of models where each model has a different number of layers. Jasper models are denoted as Jasper bxr where b and r represent:

- b: the number of blocks
- r: the number of repetitions of each convolutional layer within a block

.. image:: jasper.png

All models have 4 common layers. There is an initial convolutional layer with stride 2 to decrease the time dimension of the speech. The other 3 layers are at the end of the network. The first layer has a dilation of 2 to increase the model's receptive field. The last two layers are fully connected layers that are used to project the final output to a distribution over characters.

Each 1D-convolutional layer consists of a convolutional operation, batch normalization, clipped relu activation, and dropout. Shown on the left.

There is a residual connection between each block which consists of a projection layer, followed by batch normalization. The residual is then added to the output of the last 1D-convolutional layer in the block before the clipped relu activation and dropout. Shown on the right.

.. image:: jasper_layers.png

We preprocess the speech signal by sampling the raw audio waveform of the signal using a sliding window of 20ms with stride 10ms. We then extract log-mel filterbank energies of size 64 from these frames as input features to the model.

We use Connectionist Temporal Classification (CTC) loss to train the model. The output of the model is a sequence of letters corresponding to the speech input. The vocabulary consists of all alphabets (a-z), space, and the apostrophe symbol, a total of 29 symbols including the blank symbol used by the CTC loss.

Training
~~~~~~~~

Our current best WER is a 54 layer model trained using synthetic data and using dense residual connections following the `DenseNet paper <https://arxiv.org/abs/1608.06993>`_. We achieved a WER of 4.10% on the librispeech test-clean dataset using greedy decoding:

+----------------------------+-----------------------------------------------------------------------+
| Model                      | LibriSpeech Dataset                                                   |
+                            +-----------------+-----------------+-----------------+-----------------+
|                            | Dev-Clean       |       Dev-Other |      Test-Clean |      Test-Other |
+                            +--------+--------+--------+--------+--------+--------+--------+--------+
|                            | Greedy |  Beam  | Greedy |  Beam  | Greedy |  Beam  | Greedy |  Beam  |
+============================+========+========+========+========+========+========+========+========+
| Jasper 10x3                | 5.10   | 4.37   | 15.49  | 13.46  | 5.10   | 5.14   | 16.21  | 14.35  |
+----------------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| Jasper 10x5                | 4.51   | 3.77   | 13.88  | 12.20  | 4.59   | 4.46   | 14.34  | 12.79  |
+----------------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| Jasper 10x5 syn            | 4.32   | 3.74   | 13.74  | 11.57  | 4.32   | 4.39   | 14.08  | 12.21  |
+----------------------------+--------+--------+--------+--------+--------+--------+--------+--------+
| Jasper 10x5 dense res syn  | 4.15   | 3.64   | 13.40  | 11.37  | 4.10   | 4.04   | 14.04  | 12.40  |
+----------------------------+--------+--------+--------+--------+--------+--------+--------+--------+

We used Open SLR language model while decoding with beam search using a beam width of 128.

The models were trained for 400 (200 for syn) epochs on 8 GPUs. We use:

* SGD with momentum = 0.9
* a learning rate with polynomial decay using an initial learning rate of 0.05
* Layer-wise Adative Rate Control (LARC) with eta = 0.001
* weight-decay = 0.001
* dropout (varible per layer: 0.2-0.4)

Synthetic Data
~~~~~~~~~~~~~~
All models with "syn" in their name are trained using a combined dataset of Librispeech and synthetic data.

The training details can be found :ref:`here <synthetic_data>`.

Mixed Precision
~~~~~~~~~~~~~~~

To use mixed precision (float16) during training we made a few minor changes to the model. Tensorflow by default calls Keras Batch Normalization on 3D input (BxTxC) and cuDNN on 4D input (BxHxWxC). In order to use cuDNN's BN we added an extra dimension to the 3D input to make it a 4D tensor (BxTx1xC).

We also use backoff loss scaling.
