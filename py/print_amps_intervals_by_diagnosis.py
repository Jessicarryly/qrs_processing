import sys

import os
os.chdir("/Users/shushu/Documents/WFDB/")

def get_differencies(patient_record, output_path, signal_index):
  record_filename = "ptbdb/" + patient_record
  output_filename = output_path + "/" + "_".join(patient_record.split("/")) + "_" + signal_index
  
  with open(record_filename + "_qrs_corrected_" + str(signal_index)) as inp, \
       open(output_filename, "w") as outp:
    inp.readline()
    for line in inp:
      sample, amplitude, interval = line.strip().split("\t")[:3]
      print >> outp, "%s\t%s" % (amplitude, interval)
      prev_amplitude, prev_interval = amplitude, interval

    
patient_record = sys.argv[1]
output_path = sys.argv[2]
signal_index = sys.argv[3]


get_differencies(patient_record, output_path, signal_index)  
