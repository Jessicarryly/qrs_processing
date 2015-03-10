from base_functions import changeSFreq
import os

def resampleAllData(arduino_path, output_suffix):
  for filename in os.listdir(arduino_path):
    print filename
    splitted_filename = filename.strip().split("_")
    if len(splitted_filename) == 2:
      in_sfreq = splitted_filename[1]

      changeSFreq(arduino_path + filename, 
                  arduino_path + filename + output_suffix, 
                  int(in_sfreq), 
                  multiplier = 1000,
                  first_signal_index = 0)


if __name__ == "__main__":
  arduino_path = "/Volumes/WD500GB/WFDB/arduino/"
  output_suffix = "_resamp"
  resampleAllData(arduino_path, output_suffix)
