import sys
import numpy as np

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def robust_std(median, values):
  values = np.array(values)
  return np.sqrt(np.median(abs(values - median) ** 2))


def get_differencies(patient_record, stats_filename, output_path, signal_index):
  record_filename = "ptbdb/" + patient_record
  output_filename = output_path + "/" + "_".join(patient_record.split("/"))
  noise_filename = record_filename + "_noise"
  

  with open(record_filename + "_qrs_" + str(signal_index) + "_corrected_" + str(signal_index)) as inp, \
       open(output_filename, "w") as outp, \
       open(stats_filename, "a") as stats, \
       open(noise_filename) as noise_file:
    inp.readline()
    prev_amplitude, prev_interval = inp.readline().strip().split("\t")[1 : 3]
    print >> outp, "%s\t%s" % (prev_amplitude, prev_interval)
    amplitude_diffs = [] 
    interval_diffs = []

    for line in inp:
      sample, amplitude, interval = line.strip().split("\t")[:3]
      amplitude_diffs.append(int(amplitude) - int(prev_amplitude))
      interval_diffs.append(int(interval) - int(prev_interval))
      print >> outp, "%s\t%s" % (amplitude, interval)
      prev_amplitude, prev_interval = amplitude, interval

    noise, r_peaks_cnt = noise_file.readline().strip().split("\t")
    noise = float(noise)
    amp_diff_std = np.std(amplitude_diffs)
    noise_to_amp_diff_std = noise / (amp_diff_std + 1)
    
    print >> stats, "%s\t%.4f\t%s\t%.4f\t%.4f\t%d\t%.4f" % (patient_record,
                                                        noise_to_amp_diff_std, 
                                                        r_peaks_cnt,
                                                        amp_diff_std,
                                                        noise, 
                                                        np.mean(amplitude_diffs),
                                                        np.std(interval_diffs[1:]))

    

patient_record = sys.argv[1]
stats_filename = sys.argv[2]
output_path = sys.argv[3]
signal_index = sys.argv[4]
wrong_records = map(lambda x: x.split(" ")[0].strip(), open("need_to_be_improved").readlines())

if patient_record not in wrong_records:
  get_differencies(patient_record, stats_filename, output_path, signal_index)  
