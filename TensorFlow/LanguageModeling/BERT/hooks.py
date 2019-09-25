import tensorflow as tf
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.framework import ops
from tensorflow import GraphKeys
import numpy as np
import csv
import os
import shutil

class DumpingTensorHook(basic_session_run_hooks.LoggingTensorHook):
  def __init__(self, prefixes, output_dir="./output", exclude_keywords=[]):
    self.prefixes = prefixes
    self.output_dir = output_dir
    self.exclude_keywords = exclude_keywords
    super(DumpingTensorHook, self).__init__(tensors=[], every_n_iter=1)

  def has_prefix(self, tensor_name):
    for prefix in self.prefixes:
      if tensor_name.startswith(prefix):
        return True
    return False

  def has_exclude_keywords(self, tensor_name):
    for keyword in self.exclude_keywords:
      if keyword in tensor_name:
        return True
    return False

  def begin(self):

    def get_outputs():
      for op in tf.get_default_graph().get_operations():
        # tensor types to exclude
        for tensor in op.outputs:
          if tensor.dtype in {tf.variant, tf.string}:
            continue
          # if self.has_prefix(tensor.name) and not self.has_exclude_keywords(tensor.name):
          # if tensor.name.endswith('xxx'):
          if "probe" in tensor.name:
            yield tensor.name

    self._tensors = {item: item for item in get_outputs()}
    super(DumpingTensorHook, self).begin()

  def _log_tensors(self, tensor_values):
    # tensor_values is a map
    print('%6i being dump' %
                  (self._iter_count))
    if self._iter_count > 5:
      return
    if os.path.isdir(self.output_dir):
    #   shutil.rmtree(self.output_dir)
        pass
    else:
        os.makedirs(self.output_dir)
    
    rows = []
    for k, v in tensor_values.items():
        filepath = 'iter_{}/{}'.format(self._iter_count, k)
        filepath = os.path.join(self.output_dir, filepath)
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
            np.save(filepath, v)
            rows.append(['{}.npy'.format(filepath), v.shape])

    if self._iter_count == 0:
      with open(os.path.join(self.output_dir, 'tf_bert_tensors.csv'), 'w') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        for r in rows:
            writer.writerow(r)
      print("tensor index written")
