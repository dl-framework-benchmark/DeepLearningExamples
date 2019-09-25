import re
import argparse

parser = argparse.ArgumentParser(description="compute the average throughput rate")
parser.add_argument(dest='file', type=str, help='input log file')
args = parser.parse_args()
result = []
with open(args.file, "r") as fp:
    line = fp.readline()
    while line:
        if re.match(r'INFO:tensorflow:examples/sec: (.*)', line):
            m = re.match(r'INFO:tensorflow:examples/sec: (.*)', line)
            result.append(float(m.group(1)))
        line = fp.readline()

average = sum(result) / len(result)
print("{} samples/sec".format(average))

