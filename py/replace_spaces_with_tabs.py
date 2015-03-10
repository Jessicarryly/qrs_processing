import re 
import sys

from base_functions import str_of_list, get_splitted_line, get_signal_values

import os
os.chdir("/Users/shushu/Documents/WFDB/")


patient_record = sys.argv[1].strip()

multiplier = 1.0  
if len(sys.argv) > 2:  
  multiplier = float(sys.argv[2])


path = "/Volumes/WD500GB/WFDB/ptbdb/"
filename = path + patient_record
raw_filename = filename + "_copy"

with open(filename, "w") as outp, open(raw_filename) as inp:
  line = inp.readline()
  line = get_splitted_line(line) 
  elements_count = len(line)

  while True:
    signal_values = get_signal_values(line)
    if multiplier != 1.0:
       signal_values = signal_values * multiplier
    print >> outp, "%s\t%s" % (line[0], str_of_list(signal_values))

    line = inp.readline()
    line = get_splitted_line(line) 
    if len(line) < elements_count:
      break
