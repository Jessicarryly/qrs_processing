import re
from subprocess import call, Popen

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def print_length():
  with open("record.txt") as inp, \
       open("lengths.txt", "a") as outp:
    record = ""
    for line in inp:
      if len(record) == 0:
        record = line.split()[1]  
      if line.split(":")[0].strip() == "Length":
        print >> outp, "%s\t%s" % (record, line.split("(")[1].split()[0])

def get_description():
  with open("all_records.txt") as all_records:
    for line in all_records:
      with open("record.txt", "w") as record:
        #print >> record, call(["wfdbdesc", "ptbdb/" + line.strip()])
        x = Popen(["wfdbdesc", "ptbdb/" + line.strip()])
        print_length()

#get_description()
print_length()  
