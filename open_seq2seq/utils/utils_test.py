# Copyright (c) 2018 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import tempfile
import copy
import numpy as np
import numpy.testing as npt

from open_seq2seq.test_utils.test_speech_config import base_params, \
                                                       train_params, \
                                                       eval_params, \
                                                       base_model
# from open_seq2seq.utils.utils import get_batches_for_epoch


# class UtilsTests(tf.test.TestCase):
#   def setUp(self):
#     base_params['logdir'] = tempfile.mktemp()
#     self.train_config = copy.deepcopy(base_params)
#     self.eval_config = copy.deepcopy(base_params)
#     self.train_config.update(copy.deepcopy(train_params))
#     self.eval_config.update(copy.deepcopy(eval_params))
#
#   def tearDown(self):
#     pass
#
#   def test_get_batches_for_epoch(self):
#     length_list = []
#     num_gpus = 2
#     for bs in [1, 2, 3, 5, 7]:
#       if bs * num_gpus > 10:
#         continue
#       with tf.Graph().as_default() as g:
#         self.train_config['batch_size_per_gpu'] = bs
#         self.train_config['num_gpus'] = num_gpus
#         model = base_model(params=self.train_config, mode="train", hvd=None)
#         model.compile()
#
#         with self.test_session(g, use_gpu=True) as sess:
#           sess.run(tf.global_variables_initializer())
#           inputs_per_batch, _ = get_batches_for_epoch(model, sess, False)
#           length_list.append(np.hstack([inp[1] for inp in inputs_per_batch]))
#
#     for i in range(len(length_list) - 1):
#       npt.assert_allclose(length_list[i], length_list[i + 1])


if __name__ == '__main__':
  tf.test.main()
