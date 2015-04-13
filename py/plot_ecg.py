import sys
import os
from matplotlib import pyplot as plt
import numpy as np
from collections import defaultdict
from base_functions import getSplittedLine

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def plotECG(patient_record, info, data_path, plots_path, signal_index = 0, is_limited = False, xmin = None, xmax = None):
  data = np.loadtxt(data_path + patient_record)
  qrs_complexes = np.loadtxt(data_path + patient_record + "_qrs_corrected_" + str(signal_index), skiprows = 1, usecols = range(4))
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
  f.savefig(plots_path + str(signal_index) + "/" + \
            ";".join(patient_record.split("/")) + ";" + str(signal_index) + ".png")

def getPatientsDiagnoses(local_path):
  patients_to_diagnosis_map = {}
  for diagnosis in open(local_path + "diagnoses_list.txt"):
    diagnosis = diagnosis.strip()
    
    for patient_record in open(local_path + "diagnosis/" + diagnosis):
      patient = patient_record.strip().split("/")[0]
      patients_to_diagnosis_map[patient] = diagnosis
  return patients_to_diagnosis_map

def getRecordsWithErrorsInQRSPeaks(log_path, errors_prefix, timestamp, signal_index):
  records = [] 
  if errors_prefix:
    errors_log = log_path + errors_prefix + "_log_" + timestamp + "_" + str(signal_index)
    with open(errors_log) as src:
      for line in src:
        record = getSplittedLine(line)[0]
        records.append(record)
  return records

def plotLog(timestamp, errors_prefix = "", mode = "muv"):
  local_path = "/Users/shushu/Documents/WFDB/"
  hd_path = "/Volumes/WD500GB/WFDB/"
  log_path = local_path + "log/good_data_log/"
  data_path = hd_path + "ptbdb/"
  plots_path = hd_path + "plots/"

  if mode != "adc":
    data_path = hd_path + "ptbdb_muv/"
    plots_path = hd_path + "plots_muv/"
  
  if errors_prefix:
    plots_path += errors_prefix + "/"
    print plots_path
  
  patients_to_diagnosis_map = getPatientsDiagnoses(local_path)
  
  log = log_path + "log_" + timestamp
  for signal_index in [2, ]:
    signal_log = log + "_" + str(signal_index)
    errors_records = getRecordsWithErrorsInQRSPeaks(log_path, errors_prefix, timestamp, signal_index)

    for line in open(signal_log):
      line = getSplittedLine(line)
      patient_record = line[0].split("=")[1]
      patient, record = patient_record.split("/")
      if patient in patients_to_diagnosis_map and (not errors_records or patient_record in errors_records):
        print patient_record
        diagnosis = patients_to_diagnosis_map[patient]
        qrs_count = str(len(open(data_path + patient_record + "_qrs_" + str(signal_index)).readlines()))
      
        info = "; ".join(line[1:]) + ";TotalQRS=" + qrs_count
        plotECG(patient_record, 
                 diagnosis + "; " + info, 
                 data_path,
                 plots_path,
                 signal_index = signal_index,
                 is_limited = True, xmin = 2000, xmax = 8000)
 
def printReady():
  hd_path = "/Volumes/WD500GB/WFDB/"
  data_path = hd_path + "ptbdb_muv/"
  plots_path = hd_path + "plots_muv/"
  with open(hd_path + "ready") as src:
    for line in src:
      line = getSplittedLine(line)
      patient_record = line[0] + "/" + line[1]
      info = "diagnosis " + line[-1]
      signal_index = line[2]
      plotECG(patient_record, 
              info, 
              data_path,
              plots_path,
              signal_index = int(signal_index),
              is_limited = True, 
              xmin = 10000, 
              xmax = 25000)



if __name__ == "__main__":
  timestamp = sys.argv[1] 
  try:
    errors_prefix = sys.argv[2]
  except:
    errors_prefix = ""

  plotLog(timestamp, errors_prefix)
  
  #printReady()
  

  #plotLog("20150327_0017", "omitted")
