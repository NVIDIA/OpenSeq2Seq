# Copyright (c) 2018 NVIDIA Corporation
'''
This file implements function to calcuate basic metrics.
'''
import numpy as np
import tensorflow as tf

def true_positives(labels, preds):
  return np.sum(np.logical_and(labels, preds)) 

def accuracy(labels, preds):
  return np.sum(np.equal(labels, preds)) / len(preds)

def recall(labels, preds):
  return true_positives(labels, preds) / np.sum(labels)

def precision(labels, preds):
  return true_positives(labels, preds) / np.sum(preds)

def f1(labels, preds):
  rec = recall(labels, preds)
  pre = precision(labels, preds)
  if rec == 0 or pre == 0:
    return 0
  return 2 * rec * pre / (rec + pre)
