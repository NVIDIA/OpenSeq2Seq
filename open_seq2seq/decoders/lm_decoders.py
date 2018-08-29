# Copyright (c) 2018 NVIDIA Corporation
"""This module defines various fully-connected decoders (consisting of one
fully connected layer).

These classes are usually used for models that are not really
sequence-to-sequence and thus should be artificially split into encoder and
decoder by cutting, for example, on the last fully-connected layer.
"""
from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals
from six.moves import range

from .decoder import Decoder


class FakeDecoder(Decoder):
  """Fakde decoder for LM
  """
  def __init__(self, params, model,
               name="fake_decoder", mode='train'):
    super(FakeDecoder, self).__init__(params, model, name, mode)

  def _decode(self, input_dict):
    """This method performs linear transformation of input.

    Args:
      input_dict (dict): input dictionary that has to contain
          the following fields::
            input_dict = {
              'encoder_output': {
                'outputs': output of encoder (shape=[batch_size, num_features])
              }
            }

    Returns:
      dict: dictionary with the following tensors::

        {
          'logits': logits with the shape=[batch_size, output_dim]
          'outputs': [logits] (same as logits but wrapped in list)
        }
    """
    # return {'logits': input_dict['encoder_output']['logits'], 
    #         'outputs': [input_dict['encoder_output']['outputs']]}
    # if 'logits' in input_dict['encoder_output']:
    #   return {'logits': input_dict['encoder_output']['logits'], 
    #           'outputs': [input_dict['encoder_output']['outputs']]}
    # else:
    #   return {}
    return input_dict['encoder_output']