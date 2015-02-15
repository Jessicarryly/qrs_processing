import sys
import re


import os
os.chdir("/Users/shushu/Documents/WFDB/")


def get_diagnosis(filename):
  diagnosis = ""
  reason = ""
  with open(filename) as f:
    for line in f:
      if "Diagnose" in line:
        diagnosis = line.split(":")[1].strip()
      if "Reason for admission" in line:
        reason = line.split(":")[1].strip()
      if len(diagnosis) + len(reason) > 0:
        break
    return diagnosis, reason

def filter_one_type_diagnosis(patient):
  diagnosis, reason = get_diagnosis("header.txt")
  with open("diagnosis/" + "_".join(reason.lower().split()), "a") as output:
    print >> output, patient

def get_diagnosis_by_reason(patient):
  diagnosis, reason = get_diagnosis("header.txt")
  with open("diagnosis/" + "_".join(reason.lower().split()) + "_diagn" , "a") as output:
    print >> output, "%s\tA\t%s" % (patient, diagnosis)

patient = sys.argv[1]
filter_one_type_diagnosis(patient)

#get_diagnosis_by_reason(patient)
