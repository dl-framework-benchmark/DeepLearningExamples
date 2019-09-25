"""Convert tensorflow checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import argparse
import tensorflow as tf
import numpy as np
import os

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--tf_checkpoint_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path the TensorFlow checkpoint path.")
parser.add_argument("--of_dump_path",
                    default = None,
                    type = str,
                    required = True,
                    help = "Path to the output OneFlow model.")

args = parser.parse_args()

def _SaveWeightBlob2File(blob, folder, var):
    #print(blob.shape, blob.dtype , folder, var)
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, var)
    f = open(filename, 'wb')
    f.write(blob.tobytes())
    f.close()
    np.save(filename, blob)

def convert():
    path = args.tf_checkpoint_path
    init_vars = tf.train.list_variables(path)
    dstset = set()
    for name, shape in init_vars:
        array = tf.train.load_variable(path, name)
        #print("{};{};{};".format(name, shape, array.dtype))
        #print("Numpy array shape {}".format(array.shape))
        sep = name.rfind('/')
        blob_name = name[sep + 1:]
        op_name = name[:sep]
        op_name = op_name.replace('/', '-')
        if blob_name in ['adam_m', 'adam_v']:
            if op_name.endswith('beta') :
                op_name = op_name + '-' + blob_name[-1]
                blob_name = "beta"
            if op_name.endswith('gamma'):
                op_name = op_name + '-' + blob_name[-1]
                blob_name = "gamma"
            if op_name.endswith('bias'):
                op_name = op_name + '-' + blob_name[-1]
                blob_name = "bias"
            if op_name.endswith('embeddings'):
                op_name = op_name + '-' + blob_name[-1]
                blob_name = "weight"
            elif op_name.endswith('kernel'):
                op_name = op_name[:-6] + 'weight' + '-' + blob_name[-1]
                blob_name = "weight"
            elif op_name.endswith('output_weights'):
                op_name = op_name + '-' + blob_name[-1]
                blob_name = "weight"
            elif op_name.endswith('weights'):
                blob_name = "weight"
            elif op_name.endswith('gamma'):
                blob_name = "gamma"
        elif blob_name.endswith('embeddings'):
            op_name = op_name + '-' + blob_name
            blob_name = 'weight'
        elif blob_name in ["beta", "gamma", "bias"]:
            op_name = op_name + '-' + blob_name
        elif blob_name == "global_step":
            op_name = "global_step"
        elif blob_name == 'output_weights':
            op_name = op_name + '-' + blob_name
            blob_name = 'weight'
        elif blob_name == 'kernel':
            array = np.transpose(array)
            op_name = op_name + "-weight"
            blob_name = 'weight'
        elif blob_name in ['output_bias', 'output_weight']:
            op_name = op_name + '-' + blob_name
            blob_name = blob_name[7:]
        else:
            raise(Exception(name))
        
        checkpath = "/home/caishenghang/dataset/model_save_snapshots/snapshot_1"
        checkfname = os.path.join(checkpath, op_name, blob_name)
        if checkfname in dstset:
            raise(Exception("dst" + fname + " already converted. original name: " + name))
        else:
            dstset.add(checkfname)
        if not os.path.isfile(checkfname):
            print('before', name)
            print('after ', op_name + ' ' + blob_name)
        folder = os.path.join(args.of_dump_path, op_name)
        _SaveWeightBlob2File(array, folder, blob_name)


if __name__ == "__main__":
    convert()

