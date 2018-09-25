.. _sentiment_analysis:

Sentiment Analysis
================

######
Models
######

The model we use for sentiment analysis is the same one we use for the LSTM language model, except that the last output dimension is the number of sentiment classes instead of the vocabulary size. This sameness allows the sentiment analysis model to use the model pretrained on the language model for this task. You can choose to train the sentiment analysis task from scratch, or from the pretrained language model.

In this model, each source sentence is run through the LSTM cells. The last hidden state at the end of the sequence is then passed into the output projection layer before softmax is performed to get the predicted sentiment. If the parameter ``use_cell_state`` is set to True, the last cell state at the end of the sequence is concatenated to the last hidden state.

The datasets we currently support include SST (Stanford Sentiment Treebank) and IMDB reviews.

.. list-table::
   :widths: 1 2 1 
   :header-rows: 1

   * - Model description
     - Config file
     - Checkpoint
   * - :doc:`IDMB`
     - `imdb-wkt103.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.10-dev/example_configs/transfer/imdb-wkt103.py>`_
     - Accuracy=?
   * - :doc:`SST`
     - `sst-wkt2.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.10-dev/example_configs/transfer/sst-wkt2.py>`_
     - Accuracy=?

The model specification and training parameters can be found in the corresponding config file.

.. toctree::
   :hidden:
   :maxdepth: 1

   language-model/lstmlm

################
Getting started 
################


********
Get data
********

The SST (Stanford Sentiment Treebank) dataset contains of 10,662 sentences, half of them positive, half of them negative. These sentences are fairly short with the median length of 19 tokens. You can download the pre-processed version of the dataset `here <https://github.com/NVIDIA/sentiment-discovery/tree/master/data/binary_sst>`. The pre-processed dataset contains the files `train.csv`, `valid.csv`, `test.csv`. The dalay layer used to process this dataset is called SSTDataLayer.

The IMDB Dataset contains 50,000 labeled samples of much longer length. The median length is 205 tokens. Half of them are deemed positive and the other half negative. The train set, which contains of 25,000 samples, is separated into a train set of 24,000 samples and a validation set of 1,000 samples. The dalay layer used to process this dataset is called SSTDataLayer. The dataset can be downloaded `here <http://ai.stanford.edu/~amaas/data/sentiment/>`.

If you want to use a trained language model for this task, make sure that your dataset is processed in the same way the dataset used for the language model was. 

********
Training
********

Next let's create a simple LSTM language model by defining a config file for it or using one of the config files defined in ``example_configs/transfer``.  

* if you want to use a pretrained language model, specify the location of the pretrained language model using the parameter ``load_model``.
* change ``data_root`` to point to the directory containing the raw dataset used to train your language model, for example, the IMDB dataset downloaded above.
* change ``processed_data_folder`` to point to the location where you want to store the processed dataset. If the dataset has been pre-procesed before, the data layer can just load the data from this location.
* update other hyper parameters such as number of layers, number of hidden units, cell type, loss function, learning rate, optimizer, etc. to meet your needs. 
* choose ``dtype`` to be ``"mixed"`` if you want to use mixed-precision training, or ``tf.float32`` to train only in FP32.

For example, your config file is ``lstm-wkt103-mixed.py``. To train without Horovod, update ``use_Horovod`` to False in the config file and run::

    python run.py --config_file=example_configs/transfer/imdb-wkt2.py --mode=train_eval --enable_logs

When training with Horovod, use the following command::

    mpiexec --allow-run-as-root -np <num_gpus> python run.py --config_file=example_configs/transfer/imdb-wkt2.py --mode=train_eval --enable_logs

Some things to keep in mind:

* Don't forget to update ``num_gpus`` to the number of GPUs you want to use.
* If your GPUs run out of memory, reduce the ``batch_size_per_gpu``
parameter.

*************
Inference
*************

Running in the mode ``eval`` will evaluate your model on the evaluation set::

    python run.py --config_file=example_configs/transfer/imdb-wkt2.py --mode=eval --enable_logs


Running in the mode ``infer`` will evaluate your model on the test set::

    python run.py --config_file=example_configs/transfer/imdb-wkt2.py --mode=test --enable_logs

The performance of the model is reported on accuracy and F1 scores.
