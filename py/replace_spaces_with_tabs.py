import re 
import sys

import os
os.chdir("/Users/shushu/Documents/WFDB/")


patient_record = sys.argv[1].strip()

path = "/Volumes/WD500GB/WFDB/ptbdb/"
filename = path + patient_record
raw_filename = filename + "_copy"
#os.rename(filename, raw_filename)
with open(filename, "w") as outp, open(raw_filename) as inp:
  line = inp.readline()
  line = re.split("\s+", line.strip())
  elements_count = len(line)

  while True:
    print >> outp, "\t".join(line)
    line = inp.readline()
    line = re.split("\s+", line.strip())
    if len(line) < elements_count:
      break
