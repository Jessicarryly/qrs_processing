import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def plot_ecg(patient_record, info, folder, signal_index = 0, is_limited = False, xmin = None, xmax = None):
  data = np.loadtxt("/Volumes/WD500GB/WFDB/ptbdb/" + patient_record)
  
  qrs_complexes = np.loadtxt("ptbdb/" + patient_record + "_qrs_corrected_" + str(signal_index), skiprows = 1)
  r_samples = np.array(map(int, qrs_complexes[:, 0])) + 1
  argmin_of_qs_samples = np.array(map(int, qrs_complexes[:, 3])) + 1
  
  plt.close('all') 
  f, (ax1, ax2) = plt.subplots(2, sharey = False, figsize = (18, 8))

  ax1.plot(data[:, 0], data[:, 1 + signal_index])
  ax1.plot(r_samples, data[np.array(r_samples), 1 + signal_index], 'ro')
  ax1.plot(argmin_of_qs_samples, data[np.array(argmin_of_qs_samples), 1 + signal_index], 'go')
  ax1.set_title(info)

  if is_limited:
    ax2.plot(data[:, 0], data[:, 1 + signal_index])
    ax2.plot(r_samples, data[r_samples, 1 + signal_index], 'ro')
    ax2.plot(argmin_of_qs_samples, data[argmin_of_qs_samples, 1 + signal_index], 'go')
    ax2.set_xlim((xmin, xmax)) 
    ax2.set_ylim(min(data[xmin:xmax, 1 + signal_index]) - 30, max(data[xmin:xmax, 1 + signal_index]) + 30)
  f.savefig("/Volumes/WD500GB/WFDB/" + folder + "/" + str(signal_index) + "/" + \
            ";".join(patient_record.split("/")) + ";" + str(signal_index) + ".png")


def plot_log(log, folder):
  os.chdir("/Users/shushu/Documents/WFDB")
  patients_to_diagnosis_map = {}
  for diagnosis in open("diagnoses_list.txt"):
    diagnosis = diagnosis.strip()
    
    for patient_record in open("diagnosis/" + diagnosis):
      patient = patient_record.strip().split("/")[0]
      patients_to_diagnosis_map[patient] = diagnosis
  
  wrong_records = map(lambda x: x.split(" ")[0].strip(), open("need_to_be_improved").readlines())

  for signal_index in xrange(3):
    signal_log = log + "_" + str(signal_index)
    for line in open(signal_log):
      line = line.strip().split("\t")
      patient_record = line[0].split("=")[1]
      patient, record = patient_record.split("/")
      if patient in patients_to_diagnosis_map:
        print patient_record
        diagnosis = patients_to_diagnosis_map[patient]
        qrs_count = str(len(open("ptbdb/" + patient_record + "_qrs_" + str(signal_index)).readlines()))
        #noise_length = open("ptbdb/" + patient_record + "_noise_" + str(signal_index)).readline().strip()
        #if patient_record not in wrong_records:
        if True:
          info = "; ".join(line[1:]) + ";TotalQRS=" + qrs_count
          plot_ecg(patient_record, 
                   diagnosis + "; " + info, 
                   folder,
                   signal_index = signal_index,
                   is_limited = True, xmin = 2000, xmax = 8000)
       
if __name__ == "__main__":
  log = sys.argv[1] 
  folder = sys.argv[2]
  plot_log(log, folder)

