import os
os.chdir("/Users/shushu/Documents/WFDB/")


patients_to_est_noise = [line.strip() for line in open("recs_to_est_noise")]

errors_path = "/Users/shushu/Documents/WFDB/diagnosis/clf_errors/"
diagnoses = ["healthy", "sick"]


class Patient(object):
  def __init__(self, patient, rf_err, pca_err):
    self.patient, self.rf_err, self.pca_err = patient, rf_err, pca_err


for diagnosis in diagnoses:
  with open(errors_path + diagnosis + "_patients") as patients_inp, \
       open(errors_path + diagnosis + "_plain_errors_rf") as errors_inp, \
       open(errors_path + diagnosis + "_plain_errors_pca") as rca_errors_inp, \
       open(errors_path + diagnosis + "_errors", "w") as outp, \
       open(errors_path + diagnosis + "_errors_sorted", "w") as outp2:
    patients_errs = []

    for line in patients_inp:
      rf_error = int(errors_inp.readline().strip())
      pca_error = int(rca_errors_inp.readline().strip())

      line = line.strip()
      patient, record = line[:10], line[11:]
      patient = patient + "/" + record

      #if patient in patients_to_est_noise:
      print >> outp, "%s\t%d\t%d" % (patient, rf_error, pca_error) 
      patients_errs.append(Patient(patient, rf_error, pca_error))

    patients_errs = sorted(patients_errs, key = lambda x: x.rf_err, reverse = True)
    for patient in patients_errs:
      print >> outp2, "%s\t%d\t%d" % (patient.patient, patient.rf_err, patient.pca_err) 
