import re 
import sys

from base_functions import strOfList, getSplittedLine, getSignalValues

import os
os.chdir("/Users/shushu/Documents/WFDB/")


patient_record = sys.argv[1].strip()

multiplier = 1.0  
if len(sys.argv) > 2:  
  multiplier = float(sys.argv[2])


path = "/Volumes/WD500GB/WFDB/"
db = "ptbdb"
if len(sys.argv) > 3:
  db = sys.argv[3]
path += db + "/"


filename = path + patient_record
raw_filename = filename + "_copy"

with open(filename, "w") as outp, open(raw_filename) as inp:
  line = inp.readline()
  line = getSplittedLine(line) 
  elements_count = len(line) 

  while True:
    signal_values = getSignalValues(line)
    if multiplier != 1.0:
       signal_values = signal_values * multiplier
    print >> outp, "%s\t%s" % (line[0], strOfList(signal_values))

    line = inp.readline()
    line = getSplittedLine(line) 
    if len(line) < elements_count:
      break
