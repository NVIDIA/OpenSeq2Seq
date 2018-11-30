.. _speech_commands:

Speech Commands
===============


Introduction
~~~~~~~~~~~~
The ability to recognize spoken commands with high accuracy can be useful in a variety of contexts. To this end, Google recently released the Speech Commands dataset (see `paper <https://arxiv.org/abs/1804.03209>`_), which contains short audio clips of a fixed number of command words such as "stop", "go", "up", "down", etc spoken by a large number of speakers. To promote the use of the set, Google also hosted a `Kaggle competition <https://www.kaggle.com/c/tensorflow-speech-recognition-challenge>`_, in which the winning team attained a multiclass accuracy of 91%.

We started by experimenting with applying OpenSeq2Seq's existing image classification models on mel spectrograms of the audio clips and found that they worked surprisingly well. Adding data augmentation further improved the results.  


Dataset
~~~~~~~
Google released two versions of the dataset with the first version containing 65k samples over 30 classes and the second containing 110k samples over 35 classes. However, the Kaggle contest specification used only 10 of the provided classes, grouped the others as "unknown" and added "silence" for a total of 12 labels. We refer to these datasets as v1-12, v1-30 and v2, and have separate metrics for each version in order to compare to the different metrics used by other papers.

To preprocess a given version, we run ``speech_commands_preprocessing.py`` which balances the number of samples in each class by duplicating the samples in each class. This effectively grows the dataset, as we apply random transformations to each sample in the data layer. The script then separates each class into training, validation and test samples via an 80-10-10 split.

These samples are fed into the data layer, which randomly stretches and adds noise to each audio sample. These augmented samples are then converted into mel spectrograms and either randomly sliced or symmetrically padded with zeros until they are fixed dimension and square (120x120). This dataset is then cached during the first epoch in order to increase GPU utilization. 


Model Details and Results
~~~~~~~~~~~~~~~~~~~~~~~~~
The output of the data layer is a 120x120 image, which is fed into an unmodified ResNet-50 architecture consisting of several convolutional blocks. We obtained the following results by adding data augmentation and training ResNet-50 in float over 10 epochs.

.. list-table::
   :widths: 1 1 1 1
   :header-rows: 1

   * - Dataset
     - Validation Accuracy
     - Test Accuracy
     - Checkpoint

   * - v1-12
     - 94.7%
     - 94.7%
     - `link <https://drive.google.com/open?id=1NDPaUuwuL2G2ZhgBL75RHOfpIs4ewCOg>`_

   * - v1-30
     - 96.6%
     - 95.8%
     - `link <https://drive.google.com/open?id=1MwUX8EqGEjrSbxOyGHOLeOeqco-RF9vT>`_

   * - v2
     - 95.0%
     - 95.1%
     - `link <https://drive.google.com/open?id=199HYZRX2O1tWFGZYkYP_E-R8SvrHIiqn>`_

The configuration file used for all models is `here <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/image2label/speech_commands_float.py>`_. The only change between datasets is the ``dataset_version`` parameter, which should be set to one of ``v1-12``, ``v1-30`` or ``v2``.


Mixed Precision
~~~~~~~~~~~~~~~

We found that the model trains just as well in mixed precision using a constant loss scaling of 512. This roughly halves the required GPU memory.
