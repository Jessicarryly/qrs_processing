from base_functions import getSplittedLine
from collections import defaultdict
import os


def mergeLogFile(records_list, src_filename, dst_filename):
  with open(src_filename) as src, open(dst_filename, "w") as dst:
    for line in src:
      record = getSplittedLine(line)[0]
      if record.startswith("Patient="):
        record = record.strip().split("=")[1]
      if record in records_list:
        print >> dst, line.strip()

def mergeLogFiles(records_list_filename, timestamp, dst_dir = "good_data_log", signals_count = 3):
  hd_path = "/Volumes/WD500GB/WFDB/"
  log_path = "/Users/shushu/Documents/WFDB/log/"
  dst_log_path = log_path + dst_dir + "/"
  
  records_list = defaultdict(set)
  with open(hd_path + records_list_filename) as records_list_file:
    for line in records_list_file:
      line = getSplittedLine(line)
      records_list[line[-1]].add("/".join(line[:2]))
  
  for signal_index in xrange(int(signals_count)):
    signal_index = str(signal_index)
    suffix = timestamp + "_" + signal_index
    for filename in os.listdir(log_path):
      if filename.endswith(suffix):
        mergeLogFile(records_list[signal_index], log_path + filename, dst_log_path + filename)

if __name__ == "__main__":
  mergeLogFiles("ready_no_diagnosis", "20150327_0017", signals_count = 3)