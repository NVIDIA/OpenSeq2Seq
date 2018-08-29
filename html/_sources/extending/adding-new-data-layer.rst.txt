Adding new data layer
=====================

All data layers have to inherit from
:class:`DataLayer <data.data_layer.DataLayer>`
class. We recommend that you use
`tf.data <https://www.tensorflow.org/programmers_guide/datasets>`__
while implementing your data layer.

You need to implement the following methods:

0. Static methods:
   :func:`get_required_params() <data.data_layer.DataLayer.get_required_params>` and
   :func:`get_optional_params() <data.data_layer.DataLayer.get_optional_params>`
   to specify required and optional parameters correspondingly

1. :meth:`__init__(self, params, num_workers=None, worker_id=None) <data.data_layer.DataLayer.__init__>`:

   -  The ``params`` parameter should be a Python dictionary with options.
      Such as mini-batch shapes, padding, where to get the data from, etc.
   -  If you are using ``tf.data``, most of the ETL (extract-transform-load)
      logic should happen here.

2. :meth:`gen_input_tensors(self) <data.data_layer.DataLayer.gen_input_tensors>`:

   -  This method should return a list of Tensors or Placeholders in which
      new data will be fed. With each call it should return new objects
      (for multi-GPU support). In case of ``tf.data``, you can use output
      of ``tf.data.Iterator.get_next()``.
   -  For example: TODO.

3. :meth:`next_batch_feed_dict(self) <data.data_layer.DataLayer.next_batch_feed_dict>`:

   -  Most likely should return empty Python dictionary if you are using
      ``tf.data``.
   -  If you are using Placeholders in :meth:`gen_input_tensors(self) <data.data_layer.DataLayer.gen_input_tensors>`,
      this method should return corresponding feed\_dictionary.

4. :meth:`shuffle(self) <data.data_layer.DataLayer.shuffle>`:

   -  If you are not using ``tf.data`` this method should shuffle your dataset.

5. :meth:`get_size_in_samples(self) <data.data_layer.DataLayer.get_size_in_samples>`:

   -  Should return the size of your dataset in samples.
