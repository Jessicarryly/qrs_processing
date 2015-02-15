import sys
import os
import re

path = sys.argv[1]
patient, record = sys.argv[2].strip().split("/")
os.chdir(path + "/" + patient)

signals = [] 
if len(sys.argv) > 3:
  signals = map(int, sys.argv[3].strip().split("_"))

print signals
initial_record_filename = record + "_before_inverting" 
os.rename(record, initial_record_filename)
with open(initial_record_filename) as inp, \
     open(record, "w") as outp:
  for line in inp:
    line = re.split("\s+", line.strip())
    for signal in signals:
      line[1 + signal] = str(int(line[1 + signal]) * -1)
    line = "\t".join(line)
    print >> outp, line
    
