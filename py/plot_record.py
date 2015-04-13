from plot_ecg import plotECG
import sys

patient_record = sys.argv[1]
signal_index = sys.argv[2]
info = " ".join(sys.argv[3:])
plotECG(patient_record, 
        info, 
        "/Volumes/WD500GB/WFDB/ptbdb_muv/",
        "/Volumes/WD500GB/WFDB/plots_muv/",
         signal_index = int(signal_index),
         is_limited = True, xmin = 10000, 
         xmax = 25000)
