.. _tacotron-2-gst:

Tacotron 2 with Global Style Tokens
====================================

Model
~~~~~
This model extends Tacotron 2 with 
`Global Style Tokens <https://ai.googleblog.com/2018/03/expressive-speech-synthesis-with.html>`_
(see also `paper <https://arxiv.org/abs/1803.09017>`_). We differ from the published
paper in that we use Tacotron 2 from OpenSeq2Seq as opposed to Tacotron.

Model Description
~~~~~~~~~~~~~~~~~~
Tacotron 2 with Global Style Tokens adds a reference encoder to the Tacotron 2 model.
The reference encoder takes as input a spectrogram which is treated as the style
that the model should learn to match. The reference encoder is similar to the text
encoder. It first passes through a stack of convolutional layers followed by a
recurrent GRU network. We take the last state and treat that as the query vector to
an attention mechanism. The attention mechanism is the same as the one used in the
Transformer implementation. The keys and values are randomized at the beginning of training.
The output of this attention module is called the style embedding and concatenated
to the text embedding at every time step of the text embedding. This merged embedding
is then passed to the tacotron attention and decoder block to be decoded into a spectrogram.

Training
~~~~~~~~
We use global style tokens to model multi-speaker speech synthesis. Namely, we
are able to learn the speaker identities from the `MAILABS <http://www.m-ailabs.bayern/en/the-mailabs-speech-dataset/>`_
US dataset. Note: The MAILABS dataset has some files that exist in the csv, but
are not present in the dataset; see `issue 337 <https://github.com/NVIDIA/OpenSeq2Seq/issues/337>`_.

Training Instructions:
  1. Extract the dataset to a directory
  2. Change data_root inside tacotron_gst_combine_csv.py to point to where the
     dataset was extracted.
  3. Run tacotron_gst_combine_csv.py inside the scripts directory. The script
     will merge all the metadata csv files into one large train csv file.
  4. Change line 15 of `tacotron_gst.py <https://github.com/NVIDIA/OpenSeq2Seq/blob/master/example_configs/text2speech/tacotron_gst.py>`_
     such ``dataset_location`` points to where the dataset was extracted
  5. Train the model by running ``python run.py --config_file=example_configs/text2speech/tacotron_gst.py --mode=train``


Inference
~~~~~~~~~
Inference is similar to Tacotron infer, except tacotron-gst additionally
requires a style wav inside the infer csv. ``train.csv`` should contains lines
with lines in the following format::

    path/to/style.wav | UNUSED | This is an example sentence that I want to generate.


