import sys  

import os
os.chdir("/Documents/WFDB/")


def get_diagoses_of_ready_patients(ready):
  patients_to_diagnosis_map = {}
  for diagnosis in open("diagnoses_list.txt"):
    diagnosis = diagnosis.strip()
    
    for patient_record in open("diagnosis/" + diagnosis):
      patient = patient_record.strip().split("/")[0]
      patients_to_diagnosis_map[patient] = diagnosis
  with open(ready + "_no_diagnosis") as inp, \
       open(ready, "w") as outp:
    for line in inp:
      line = line.strip()
      patient = line.split(" ")[0]
      diagnosis = patients_to_diagnosis_map[patient].strip()
      print >> outp, line + " " + diagnosis.strip() 

ready = sys.argv[1].strip()
get_diagoses_of_ready_patients(ready)
