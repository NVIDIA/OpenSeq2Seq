# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import copy
import tempfile

import numpy as np
import numpy.testing as npt
import tensorflow as tf
from six.moves import range

from open_seq2seq.test_utils.test_speech_configs.ds2_test_config import \
    base_params, train_params, eval_params, base_model
from open_seq2seq.utils.utils import get_results_for_epoch, get_available_gpus


class UtilsTests(tf.test.TestCase):

  def setUp(self):
    base_params['logdir'] = tempfile.mktemp()
    self.train_config = copy.deepcopy(base_params)
    self.eval_config = copy.deepcopy(base_params)
    self.train_config.update(copy.deepcopy(train_params))
    self.eval_config.update(copy.deepcopy(eval_params))

  def tearDown(self):
    pass

  def test_get_results_for_epoch(self):
    # this will take all gpu memory, but that's probably fine for tests
    gpus = get_available_gpus()
    length_list = []
    for num_gpus in [1, 2, 3]:
      if num_gpus > len(gpus):
        continue
      for bs in [1, 2, 3, 5, 7]:
        if bs * num_gpus > 10:
          continue
        with tf.Graph().as_default() as g:
          self.eval_config['batch_size_per_gpu'] = bs
          self.eval_config['num_gpus'] = num_gpus
          model = base_model(params=self.eval_config, mode="infer", hvd=None)
          model.compile()
          model.infer = lambda inputs, outputs: inputs
          model.finalize_inference = lambda results: results

          with self.test_session(g, use_gpu=True) as sess:
            sess.run(tf.global_variables_initializer())
            inputs_per_batch = get_results_for_epoch(
                model, sess, False, "infer")
            length = np.hstack([inp['source_tensors'][1]
                                for inp in inputs_per_batch])
            ids = np.hstack([inp['source_ids'] for inp in inputs_per_batch])
            length_list.append(length[np.argsort(ids)])

    for i in range(len(length_list) - 1):
      npt.assert_allclose(length_list[i], length_list[i + 1])


if __name__ == '__main__':
  tf.test.main()
