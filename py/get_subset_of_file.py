import sys
from numpy.random import seed, permutation

import os
os.chdir("/Users/shushu/Documents/WFDB/")


input_filename = sys.argv[1]
output_filename = sys.argv[2]
subset_proportion = float(sys.argv[3])
input_seed=int(sys.argv[4])
seed(123 + input_seed)
line_count = len(open(input_filename).readlines())
subset_lines = permutation(line_count)[:int(line_count * subset_proportion + 1)]
print line_count, len(subset_lines)
with open(input_filename) as inp, \
     open(output_filename, "w") as outp:
  line_index = 0
  for line in inp:
    if line_index in subset_lines:
      print >> outp, line.strip()
    line_index += 1

