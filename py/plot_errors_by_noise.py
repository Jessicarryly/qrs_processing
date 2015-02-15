from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def load_sick_or_healthy_errors(errors_filename, patients_errors, diagnosis):
  for line in open(errors_filename):
    line = line.strip().split("\t")
    patients_errors[line[0]] = [diagnosis, int(line[1])]
  return patients_errors


def load_patients_errors():
  errors_path = "/Users/shushu/Documents/WFDB/diagnosis/clf_errors/"
  healthy_errors_filename = errors_path + "healthy_errors"
  sick_errors_filename = errors_path + "sick_errors"
  patients_errors = defaultdict(list)
  patients_errors = load_sick_or_healthy_errors(healthy_errors_filename, patients_errors, "healthy")
  patients_errors = load_sick_or_healthy_errors(sick_errors_filename, patients_errors, "sick")
  return patients_errors


class Record(object):
  def __init__(self, name, noise_to_amp_std, r_peaks_cnt, noise_std, diagnosis, clf_errors):
    self.name, self.noise_to_amp_std, self.r_peaks_cnt, self.noise_std, self.diagnosis, self.clf_errors = \
      name, noise_to_amp_std, r_peaks_cnt, noise_std, diagnosis, clf_errors


def load_records_stats():
  patients_errors = load_patients_errors()
  with open("diagnosis/stats") as stats_file:
    records_stats = []
    stats_file.readline()
    for line in stats_file:
      line = line.strip().split("\t")
      
      if line[0] in patients_errors:
        records_stats.append(Record(line[0], float(line[1]), \
                                    int(line[2]), float(line[4]), \
                                    *patients_errors[line[0]]))
  return records_stats


def plot_hists(records_stats):
  errors = [x.clf_errors for x in records_stats]
  
  f, (ax1, ax2) = plt.subplots(2, sharey = False, figsize = (18, 8))
  noises = [x.noise_to_amp_std for x in records_stats]
  freq, bins = np.histogram(noises, 
                            bins = [0.0146, 0.02752667, 0.03399, 0.04045333, 0.04691667, \
                                    0.05338, 0.05984333, 0.06630667, 0.07277, 0.07923333, \
                                    0.09216, 0.10508667, 0.13094, 0.209])
  ind_of_bins = np.digitize(noises, bins)
  bin_errors = np.zeros(len(freq))
  for ind in xrange(len(errors)):
    bin_errors[ind_of_bins[ind] - 1] += errors[ind]
  for ind in xrange(len(bin_errors)):
    bin_errors[ind] /= 1.0 * freq[ind]
  
  ax1.bar(np.arange(len(bin_errors)), bin_errors)
  ax1.set_xticks(np.arange(len(bin_errors)))
  ax1.set_xticklabels(map(lambda x : "%.3f\n%d" % x, zip(bins, freq)))

  r_peaks_cnt = [x.r_peaks_cnt for x in records_stats]
  freq, bins = np.histogram(r_peaks_cnt, 
                            bins = [39.0, 108.30, 114.075, 119.85000000000001, 125.625, 131.4, 137.175, \
                                    142.950, 148.725, 154.5, 160.275, 166.05, 177.6, 200.70, 280])
  ind_of_bins = np.digitize(r_peaks_cnt, bins)
  bin_errors = np.zeros(len(freq))
  for ind in xrange(len(ind_of_bins)):
    bin_errors[ind_of_bins[ind] - 1] += errors[ind]  
  for ind in xrange(len(bin_errors)):
    bin_errors[ind] /= 1.0 * freq[ind]
  
  ax2.bar(np.arange(len(bin_errors)), bin_errors)
  ax2.set_xticks(np.arange(len(bin_errors)))
  ax2.set_xticklabels(map(lambda x : "%.3f\n%d" % x, zip(bins, freq)))
  
  plt.savefig("errors_hist_by_noises_and_length.png")
  plt.clf()


def plot_scatter(records_stats):
  color = {
    "healthy" : 1, 
    "sick" : 2
  }
  plt.scatter([x.noise_to_amp_std for x in records_stats], \
              [x.r_peaks_cnt for x in records_stats], \
              s = [x.clf_errors * 50 for x in records_stats], 
              c = [color[x.diagnosis] for x in records_stats], 
              alpha = 0.7)
  plt.savefig("errors_scatter.png")
  plt.clf()


def plot_a_plot(records_stats):
  f, (ax1, ax2) = plt.subplots(2, sharey = False, figsize = (18, 8))

  sorted_by_noise = sorted(records_stats, key = lambda x: x.noise_to_amp_std)
  ax1.plot([x.clf_errors for x in sorted_by_noise])
  ax1.set_xticklabels([x.noise_to_amp_std for x in sorted_by_noise])
  
  sorted_by_errors = sorted(records_stats, key = lambda x: x.r_peaks_cnt)
  ax2.plot([x.clf_errors for x in sorted_by_errors])
  ax2.set_xticklabels([x.r_peaks_cnt for x in sorted_by_errors])
  plt.savefig("errors_by_noises_and_length.png")
  plt.clf()

records_stats = load_records_stats()
plot_hists(records_stats)
plot_scatter(records_stats)  
plot_a_plot(records_stats)
  
