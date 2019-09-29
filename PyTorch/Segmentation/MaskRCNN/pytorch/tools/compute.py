# -*- coding: utf-8 -*-
import argparse
import sys
import re

import numpy as np


def parse_arguements(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, help='', default='./logs/py_mask-rcnn_2n8c_n01.log')
    parser.add_argument('--output_dir', type=str, help='', default='../output')
    parser.add_argument('--model', type=str, help='', default='py_mask-rcnn')
    parser.add_argument('--total_batch', type=int, help='', default=16)
    # parser.add_argument('--max_step', type=int, help='', default=100)
    return parser.parse_args(argv)


def extract_time_per_iter(log_dir):
    time_per_iter = 0
    source = open(log_dir, 'r', encoding='UTF-8')
    lines = source.readlines()
    for line in lines:
        if ("Total training time" in line):
            p1 = re.compile(r'[(](.*?)[)]', re.S)
            time = re.findall(p1, line)
            temp=time[0].replace(" ", "")
            time_per_iter = temp.strip("s/it")
            print(time_per_iter)
    return float(time_per_iter)


def compute_throughput_rate(args):
    print("Computing throughput rate")
    time_per_iter = extract_time_per_iter(args.log_dir)
    iter_per_sec = 1 / time_per_iter
    throughput_rate = iter_per_sec * args.total_batch
    print("Throughput rate is : ", throughput_rate)


def main(args):
    print("Start to commpute:")
    compute_throughput_rate(args)


if __name__ == '__main__':
    main(parse_arguements(sys.argv[1:]))
