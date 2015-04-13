import sys
import numpy as np
from collections import defaultdict
import re

import os
os.chdir("/Users/shushu/Documents/WFDB/")


class SignalDescription(object):
  def __init__(self, signal_index, amp_median, total, intervals_outliers):
    self.signal_index = signal_index
    self.amp_median, self.total = amp_median, total
    self.intervals_outliers = intervals_outliers

  def __str__(self):
    return "%s\t%.4f\t%d\t%d" % (self.signal_index, self.amp_median, self.total, self.intervals_outliers)


class PatientRecordDescription(object):
  def __init__(self, patient_record, diagnosis):
    self.patient_record, self.diagnosis = patient_record, diagnosis
    self.selected_signal = None
    self.signal_list = []

  def __str__(self):
    pass

  def selectSignal(self, muv = True):
    signal_descriptions_initial = [self.loadSignal(signal_index, muv) for signal_index in self.signal_list]
    signal_descriptions = sorted(signal_descriptions_initial, key = lambda x: x.total, reverse = True)
    if signal_descriptions[0].total > signal_descriptions[-1].total + 5:
      signal_descriptions = signal_descriptions[:-1]

    signal_descriptions = sorted(signal_descriptions, key = lambda x: x.amp_median, reverse = True)
    self.selected_signal = signal_descriptions[0]
    return signal_descriptions_initial, map(lambda x: x.signal_index, signal_descriptions) 


  def loadSignal(self, signal_index, muv = True):
    signal_index = str(signal_index)
    path = "/Volumes/WD500GB/WFDB/ptbdb_muv/"
    if muv == False:
      path = "ptbdb/"
    qrs_input_filename = path + self.patient_record + "_qrs_corrected_" + str(signal_index)
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
    return SignalDescription(signal_index, np.median(amplitudes), len(amplitudes), len(intervals[intervals > int_mean + 3 * int_std]))


def getPatientRecords(input_filename):
  patient_records = []
  patient_records_descriptions = {}

  for line in open(input_filename):
    line = re.split("\s+", line.strip())
    patient_record = line[0] + "/" + line[1]
    if patient_record not in patient_records_descriptions:
      patient_records_descriptions[patient_record] = PatientRecordDescription(patient_record, line[3])
      patient_records.append(patient_record)
    patient_records_descriptions[patient_record].signal_list.append(line[2])
    
  return patient_records, patient_records_descriptions


def selectSignals(input_filename, print_to_log = False, muv = True):
  patient_records, patient_records_descriptions = getPatientRecords(input_filename)
  if print_to_log:
    with open("log/selected_signals_descriptions", "w") as log, \
         open("log/selected_signals_descriptions_err", "w") as err_log:
      for patient_record in patient_records:
        signal_descriptions_initial, signal_descriptions = patient_records_descriptions[patient_record].selectSignal(muv)
        
        if len(signal_descriptions) < len(signal_descriptions_initial):
          print >> err_log, patient_record, \
                      patient_records_descriptions[patient_record].selected_signal.signal_index, \
                      "\t".join(map(str, signal_descriptions_initial)), \
                      "Problemma!! ", ",".join(signal_descriptions), len(signal_descriptions_initial)

        print >> log, patient_record, \
                      patient_records_descriptions[patient_record].selected_signal.signal_index, \
                      "\t".join(map(str, signal_descriptions_initial))
  else:
    for patient_record in patient_records:
        patient_records_descriptions[patient_record].selectSignal(muv)
  return patient_records, patient_records_descriptions


def printPatientRecordsDescriptions(output_filename, patient_records, descriptions):
  with open(output_filename, "w") as output:
    for patient_record in patient_records:
      patient, record = patient_record.split("/")
      print >> output, patient, record, \
               descriptions[patient_record].selected_signal.signal_index, \
               descriptions[patient_record].diagnosis

if __name__ == "__main__":
  muv = True
  if len(sys.argv) > 1:
    muv = bool(sys.argv[1])
  input_filename = "/Volumes/WD500GB/WFDB/ready"
  patient_records, patient_record_descriptions = selectSignals(input_filename, muv)

  output_filename = input_filename + "_selected"
  printPatientRecordsDescriptions(output_filename, patient_records, patient_record_descriptions)




