import os
os.chdir("/Users/shushu/Documents/WFDB/")


wrong_records = map(lambda x: x.split(" ")[0].strip(), open("need_to_be_improved").readlines())
patients = []
with open("recs_to_est_noise", "w") as outp, open("recs") as inp:
  for line in inp:
    line = line.strip()
    patient = line[:10]
    patients.append(patient)

    rec = line[11:].strip()
    record = patient + "/" + rec
    if record not in wrong_records:
      print >> outp, patient + "/" + rec


print len(set(patients))
