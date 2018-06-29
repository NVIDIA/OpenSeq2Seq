# Copyright (c) 2017 NVIDIA Corporation
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

import tensorflow as tf
import numpy as np
import copy
import numpy.testing as npt
import tempfile
import os
import pandas as pd

from .speech2text_test import Speech2TextModelTests
from open_seq2seq.test_utils.test_speech_configs.w2l_test_config import base_params, \
                                                                        train_params, \
                                                                        eval_params, \
                                                                        base_model


class DS2ModelTests(Speech2TextModelTests):

  def setUp(self):
    self.base_model = base_model
    self.base_params = base_params
    self.train_params = train_params
    self.eval_params = eval_params

  def tearDown(self):
    pass

  def test_convergence(self):
    return self.convergence_test(5.0, 30.0, 0.1)

  def test_convergence_with_iter_size(self):
    return self.convergence_with_iter_size_test()

  def test_infer(self):
    return self.infer_test()

  def test_mp_collection(self):
    return self.mp_collection_test(14, 6)


if __name__ == '__main__':
  tf.test.main()
