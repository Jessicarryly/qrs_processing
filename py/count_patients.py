import sys

import os
os.chdir("/Users/shushu/Documents/WFDB/")


filename, diagnosis, output_filename = sys.argv[1:]

patients = []
for line in open(filename):
  line = line.strip().split("/")
  patients.append(line[0])

with open(output_filename, "a") as outp:
  print >> outp, "%s\t%d\t%d" % (diagnosis, len(set(patients)), len(patients))
  #print >> outp, "\n".join(patients)
