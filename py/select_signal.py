import sys
import numpy as np

import os
os.chdir("/Users/shushu/Documents/WFDB/")


class SignalDescription(object):
  def __init__(self, amp_median, total, intervals_outliers):
    self.amp_median, self.total = amp_median, total
    self.intervals_outliers = intervals_outliers

  def __str__(self):
    return "%.4f\t%d\t%d" % (self.amp_median, self.total, self.intervals_outliers)


def load_signal(patient, signal_index):
  try:  
    qrs_input_filename = "ptbdb/" + patient + "_qrs_corrected_" + str(signal_index)
    qrs_input = open(qrs_input_filename)
  except IOError:
    qrs_input_filename = "ptbdb/" + patient + "_qrs_" + str(signal_index) + "_corrected_" + str(signal_index)
    qrs_input = open(qrs_input_filename)
  qrs_input.readline()
  amplitudes = []
  intervals = []
  for line in qrs_input:
    line = line.strip().split("\t")
    amplitudes.append(float(line[1]))
    intervals.append(float(line[2]))
  intervals = np.array(intervals)
  int_mean, int_std = np.mean(intervals), np.std(intervals)
  return SignalDescription(np.median(amplitudes), len(amplitudes), len(intervals[intervals > int_mean + 3 * int_std]))


def select_signal(patient_record):
  signal_descriptions_initial = [load_signal(patient_record, signal_index) for signal_index in xrange(3)]
  signal_descriptions = sorted(enumerate(signal_descriptions_initial), key = lambda x: x[1].total, reverse = True)
  if signal_descriptions[0][1].total > signal_descriptions[-1][1].total + 5:
    signal_descriptions = signal_descriptions[:-1]
  #if len(signal_descriptions) == 2:
  #  print patient_record
  signal_descriptions = sorted(signal_descriptions, key = lambda x: x[1].amp_median, reverse = True)
  return signal_descriptions[0], signal_descriptions_initial


def select_for_all_patients():
  diagnosis = sys.argv[1].strip()
  prefix = sys.argv[2].strip()
  with open(prefix + "_" + diagnosis, "w") as out, \
       open(prefix + "_info_" + diagnosis, "w") as info_out:
    for patient_record in open("diagnosis/" + diagnosis.strip()):
      patient_record = patient_record.strip()
      selected_signal, signal_descriptions = select_signal(patient_record)
      print >> info_out, "%s\t%d\t%s\t%s" % (patient_record, selected_signal[0], str(selected_signal[1]),
                                        "\t".join(map(str, signal_descriptions)))  
  print >> out, "%s\t%d" % (patient_record, selected_signal[0])

#select_for_all_patients()
patient_record = sys.argv[1].strip()
print select_signal(patient_record)[0][0]
