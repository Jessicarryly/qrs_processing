import sys

import os
os.chdir("/Users/shushu/Documents/WFDB/")

def get_differencies(patient_record, output_path, signal_index, input_path):
  record_filename = input_path + patient_record
  output_filename = output_path + "/" + "_".join(patient_record.split("/"))
  
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

input_path = "ptbdb/"
if len(sys.argv) > 4 and sys.argv[4] == "MUV":
	input_path = "/Volumes/WD500GB/WFDB/ptbdb_muv/"

get_differencies(patient_record, output_path, signal_index, input_path)  
