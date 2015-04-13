import sys
import os
import re

from sqrs import getQPeaks

def invertSignalBySign(path, patient, record, signals, suffix = "_before_inverting"):
  os.chdir(path + "/" + patient)
  try:
    signals = list(signals)
  except:
    signals = [signals, ]

  initial_record_filename = record + suffix
  os.rename(record, initial_record_filename)
  with open(initial_record_filename) as inp, \
       open(record, "w") as outp:
    for line in inp:
      line = re.split("\s+", line.strip())
      for signal in list(signals):
        line[1 + signal] = str(float(line[1 + signal]) * -1)
      line = "\t".join(line)
      print >> outp, line

def getQPeaksForInverted(path, patient, record, signals, threshold):
  for signal_index in signals:
    getQPeaks(path + "/" + patient + "/", 
              record, 
              path + "/" + patient + "/resample/", 
              path + "/" + patient + "/",
              "_sqrs_output_", 
              1000, 
              threshold, 
              signal_index= signal_index, 
              ADC= False)


if __name__ == "__main__":    
  path = sys.argv[1]
  patient, record = sys.argv[2].strip().split("/")

  signals = [] 
  if len(sys.argv) > 3:
    signals = map(int, sys.argv[3].strip().split("_"))
  
  if len(sys.argv) > 4:
    if sys.argv[4] == "INV":
      invertSignalBySign(path, patient, record, signals)
  if len(sys.argv) > 5:
    threshold = float(sys.argv[5])
    getQPeaksForInverted(path, patient, record, signals, threshold)
  

