import sys
import os
import numpy as np
from collections import defaultdict
from base_functions import getSplittedLine
from plot_ecg import getRecordsWithErrorsInQRSPeaks, plotECG
from shutil import copyfile, move 
from os import remove

def copyQRSOfMUVWithErrors(timestamp):
  hd_path = "/Volumes/WD500GB/WFDB/"
  local_path = "/Users/shushu/Documents/WFDB/"
  log_path = local_path + "log/good_data_log/"
  src_path = local_path + "ptbdb/"
  dst_path = hd_path + "ptbdb_muv/"
  copy_prefix = "copy"
  log = copy_prefix + "_" + timestamp
  info = ""
  for signal_index in xrange(3):
    signal_log = log + "_" + str(signal_index)
    errors_records = getRecordsWithErrorsInQRSPeaks(log_path, copy_prefix, timestamp, signal_index)

    for errors_record in errors_records:
      #print errors_record
      qrs_corrected = errors_record + "_qrs_corrected_" + str(signal_index)
      
      try:
        f = open(dst_path + qrs_corrected + "_copy")
      except IOError:
        print errors_record
        #move(dst_path + qrs_corrected, dst_path + qrs_corrected + "_copy")
      
      try:
        f = open(dst_path + qrs_corrected)
      except IOError:
        copyfile(src_path + qrs_corrected, dst_path + qrs_corrected)
      
      plotECG(errors_record, 
              info, 
              dst_path,
              "/Volumes/WD500GB/WFDB/plots_muv/omitted/",
               signal_index = int(signal_index),
               is_limited = True, 
               xmin = 5000, 
               xmax = 20000)

   
def getCopiesOfADCFiles(src_qrs_corrected, dst_qrs_corrected, src_qrs, dst_qrs, record):
  try:
    copyfile(src_qrs_corrected, dst_qrs_corrected)
    copyfile(src_qrs, dst_qrs)
  except IOError:
    print record
    pass

def getMuvInsteadOfADC(muv_qrs, muv_filename, signal_index):
  data = np.loadtxt(muv_filename)[:, 1 + int(signal_index)]

  with open(muv_qrs + "_copy") as src, open(muv_qrs, "w") as dst:
    line = src.readline()
    print >> dst, line.strip()
    prev_rpeak = -1
    for line in src:
      line = getSplittedLine(line)
      rpeak = int(line[0])
      if prev_rpeak >= 0:
        interval = rpeak - prev_rpeak
        print >> dst, "%d\t%.1f\t%d\t%s" % (prev_rpeak, amplitude, interval, "\t".join(prev_line[3:]))

      argmin_of_qs = int(line[3])
      amplitude = float(line[1])
      if amplitude > -100000.0:
        amplitude = data[rpeak + 1] - data[argmin_of_qs + 1]  
      prev_line = line
      prev_rpeak = rpeak
      
def deleteCopies(qrs_corrected, qrs, record):
  try:
    remove(qrs_corrected)
    remove(qrs)
  except IOError:
    print record
    pass

def getQRSOfCopiedRecords(timestamp):
  hd_path = "/Volumes/WD500GB/WFDB/"
  local_path = "/Users/shushu/Documents/WFDB/"
  log_path = local_path + "log/good_data_log/"
  src_path = local_path + "ptbdb/"
  dst_path = hd_path + "ptbdb_muv/"
  copy_prefix = "copy"
  log = copy_prefix + "_" + timestamp
  for signal_index in xrange(3):
    signal_log = log + "_" + str(signal_index)
    errors_records = getRecordsWithErrorsInQRSPeaks(log_path, copy_prefix, timestamp, signal_index)
    for errors_record in errors_records:
      qrs = dst_path + errors_record + "_qrs_" + str(signal_index) 
      qrs_corrected = dst_path + errors_record + "_qrs_corrected_" + str(signal_index) 
      getCopiesOfADCFiles(src_path + errors_record + "_qrs_corrected_" + str(signal_index), qrs_corrected + "_copy", 
                          src_path + errors_record + "_qrs_" + str(signal_index), qrs + "_copy", 
                          errors_record)
      getMuvInsteadOfADC(qrs_corrected, dst_path + errors_record, signal_index)
      getMuvInsteadOfADC(qrs, dst_path + errors_record, signal_index)
      
      deleteCopies(qrs_corrected + "_copy", qrs + "_copy", errors_record)

if __name__ == "__main__":
  timestamp = "20150327_0017"
  #copyQRSOfMUVWithErrors(timestamp)
  getQRSOfCopiedRecords(timestamp)