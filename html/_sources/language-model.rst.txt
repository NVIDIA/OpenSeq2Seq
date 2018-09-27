.. _language_model:

Language Model
================

######
Models
######

Currently we support the following models:

.. list-table::
   :widths: 1 2 1 
   :header-rows: 1

   * - Model description
     - Config file
     - Checkpoint
   * - :doc:`LSTM with WikiText-2 </language-model/lstm>`
     - `lstm-wkt2-fp32.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.10-dev/example_configs/lm/lstm-wkt2-fp32.py>`_
     - `Perplexity=89.9 <https://drive.google.com/a/nvidia.com/file/d/1uP-zALrSDb_dz7r7OUqEOakEbSIkxGEp/view?usp=sharing>`_
   * - :doc:`LSTM with WikiText-103 </language-model/lstm>`
     - `lstm-wkt103-mixed.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/18.10-dev/example_configs/lm/lstm-wkt103-mixed.py>`_
     - `Perplexity=48.6 <https://drive.google.com/a/nvidia.com/file/d/1XC4oN7PXwJwR0KFgljsJQ83CDS09LnIb/view?usp=sharing>`_

The model specification and training parameters can be found in the corresponding config file.

.. toctree::
   :hidden:
   :maxdepth: 1

   language-model/lstmlm

################
Getting started 
################

The current LSTM language model implementation supports the 
`WikiText-2 and WikiText-103 <https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/>`_ datasets. For more details about the model including hyperparameters and tips, please see 
:doc:`LSTM Language Model </language-model/lstmlm>`.

********
Get data
********

The WkiText-103 dataset, developed by Salesforce, contains over 100 million tokens extracted from the set of verified Good and Featured articles on Wikipedia. It has 267,340 unique tokens that appear at least 3 times in the dataset. Since it has full-length Wikipedia articles, the dataset is well-suited for tasks that can benefit of long term dependencies, such as language modeling.

The WikiText-2 dataset is a small version of the WikiText-103 dataset as it contains only 2 million tokens. This small dataset is suitable for testing your language model.

The WikiText datasets are available in both word-level (with minor preprocessing and rare tokens being replaced with <UNK>) and the raw character level. OpenSeq2Seq's WKTDataLayer is equipped to deal with both versions, but we recommend that you use the raw dataset.

You can download the datasets `here <https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/>`, extract them to the location of your choice. The dataset should contain of 3 files for train, validation, and test. Don't forget to update the ``data_root`` parameter in your config file to point to the location of your dataset. 

WKTDataLayer does the necessary pre-processing to make the WikiText datasets ready to be fed into the model. We use the ``word_token`` method available in the ``nltk`` package. 

You can pre-process the data for your language model in any way you deem fit. However, if you want to use your trained language model for other tasks such as sentiment analysis, make sure that the dataset used for your language model and the dataset used for the sentiment analysis model have similar pre-processing and share the same vocabulary.

********
Training
********

Next let's create a simple LSTM language model by defining a config file for it or using one of the config files defined in ``example_configs/lstmlm``.  

* change ``data_root`` to point to the directory containing the raw dataset used to train your language model, for example, your WikiText dataset downloaded above.
* change ``processed_data_folder`` to point to the location where you want to store the processed dataset. If the dataset has been pre-procesed before, the data layer can just load the data from this location.
* update other hyper parameters such as number of layers, number of hidden units, cell type, loss function, learning rate, optimizer, etc. to meet your needs. 
* choose ``dtype`` to be ``"mixed"`` if you want to use mixed-precision training, or ``tf.float32`` to train only in FP32.

For example, your config file is ``lstm-wkt103-mixed.py``. To train without Horovod, update ``use_Horovod`` to False in the config file and run::

    python run.py --config_file=example_configs/lstmlm/lstm-wkt103-mixed.py --mode=train_eval --enable_logs

When training with Horovod, use the following command::

    mpiexec --allow-run-as-root -np <num_gpus> python run.py --config_file=example_configs/lstmlm/lstm-wkt103-mixed.py --mode=train_eval --use_horovod=True --enable_logs

Some things to keep in mind:

* Don't forget to update ``num_gpus`` to the number of GPUs you want to use.
* If the vocabulary is large (the word-level vocabulary for WikiText-103 is 267,000+), you might want to use ``BasicSampledSequenceLoss``, which uses sampled softnax, instead of ``BasicSequenceLoss``, which uses full softmax.
* If your GPUs still run out of memory, reduce the ``batch_size_per_gpu``
parameter.

*************
Inference
*************

Even if your training is done using sampled softmax, evaluation and text generation will always be done using full softmax. Running in the mode ``eval`` will evaluate your model on the evaluation set::

    python run.py --config_file=example_configs/lstmlm/lstm-wkt103-mixed.py --mode=eval --enable_logs


Running in the mode ``infer`` will generate text from the seed tokens, defined in the config file under the parameter name ``seed_tokens``, each seed token should be separated by space. [TODO: make ``seed_tokens`` take a list of strings instead]::

    python run.py --config_file=example_configs/lstmlm/lstm-wkt103-mixed.py --mode=infer --enable_logs
