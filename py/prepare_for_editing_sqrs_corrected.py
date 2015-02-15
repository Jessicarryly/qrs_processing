import re

import os
os.chdir("/Users/shushu/Documents/WFDB/")


hd_path = "/Volumes/WD500GB/WFDB/"
local_path = os.getcwd() + "/ptbdb/"

edit_type = "invert"

with open(hd_path + "to_" + edit_type) as inp:
	hd_path += "ptbdb/"
	for line in inp:
		line = line.strip().split(" ")
		signal = line[2][0]
		patient_record  = line[0] + "/" + line[1] 
		
		sqrs_output_filename = hd_path + patient_record + "_sqrs_output_" + signal
		with open(sqrs_output_filename, "w") as sqrs_output_file:
			corrected_sqrs_input_filename = local_path + patient_record + "_qrs_corrected_" + signal + "_before_" + edit_type + "ing"
			corrected_sqrs_input_file = open(corrected_sqrs_input_filename)
			corrected_sqrs_input_file.readline()

			for corrected_qrs in corrected_sqrs_input_file:
				corrected_qrs = int(re.split("\s+", corrected_qrs.strip())[3])
				if corrected_qrs > 70:
					corrected_qrs -= 70
				elif corrected_qrs > 40:
					corrected_qrs -= 40
				else:
					corrected_qrs = 0
				print >> sqrs_output_file, "%d\t%d" % (corrected_qrs, corrected_qrs)



		


