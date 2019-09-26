# -*- coding: utf-8 -*-
import argparse
import sys
import re

import numpy as np


def parse_arguements(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='', default='./logs/py_bert_2n8c_n0.log')
    parser.add_argument('--output_dir', type=str, help='', default='../output')
    parser.add_argument('--model', type=str, help='', default='bert')
    parser.add_argument('--total_batch', type=int, help='', default=16)
    parser.add_argument('--num_epochs', type=int, help='', default=1)
    return parser.parse_args(argv)


def extract_iter_per_sec(log_dir):
    iter_per_sec = []
    source = open(log_dir)
    lines = source.readlines()
    for line in lines:
        if r'it/s' in line:
            # print(line)
            # 正浮点数 [0-9]*\.?[0-9]
            p1 = re.compile(r'[0-9]*\.?[0-9]+it/s', re.S)
            item = re.findall(p1, line)
            if len(item) == 0:
                continue
            num = item[0].strip('it/s')
            iter_per_sec.append(float(num))
    return iter_per_sec


def compute_throughput_rate(args):
    print("Computing throughput rate")
    iter_per_sec = np.mean(extract_iter_per_sec(args.log_dir))
    throughput_rate = iter_per_sec / args.num_epochs * args.total_batch
    print("The throughput rate is:", throughput_rate)


def main(args):
    print("Start to commpute:")
    compute_throughput_rate(args)


if __name__ == '__main__':
    main(parse_arguements(sys.argv[1:]))
