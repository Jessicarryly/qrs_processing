#!/usr/bin/python 
# -*- coding: utf-8 -*-

import numpy as np
import re
import sys
from matplotlib import pyplot as plt
from math import log, exp
from sklearn.cross_validation import train_test_split

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def get_amplitudes_intervals_stats(qrs_filename):
  with open(qrs_filename) as qrs_file:
    #первая строка файла - названия столбцов
    qrs_file.readline()
    #первый цикл не информативен по длине rr интервала, поэтому пропускаем
    #qrs_file.readline()
    amplitudes_intervals = []
    for line in qrs_file:
      amplitude_interval = line.split("\t")[-2:]
      amplitudes_intervals.append(map(int, amplitude_interval))
    medians = np.median(amplitudes_intervals, axis = 0)
    means = np.mean(amplitudes_intervals, axis = 0)
    stds = np.std(amplitudes_intervals, axis = 0)
    return medians, means, stds

def print_stats(stats_filename, records, stats):
  with open(stats_filename, "w") as stats_file:
    print >> stats_file, "Record\tAmpMed\tIntMed\tAmpStd\tIntStd\tAmpStdToMed\tIntStdToMed" 
    for i in xrange(records.shape[0]):
      print >> stats_file, records[i, 0] + "\t" + "\t".join(map(str, stats[i, 1:]))
  

def get_records_stats(patients_filename):
  with open(patients_filename) as patients_records_file:
    records = []
    records_stats = np.empty([0])
    record_index = 0
    for record in patients_records_file:
      record = record.strip()
      records.append(record)

      qrs_filename = "ptbdb/" + record + "_qrs"
      medians, means, stds = get_amplitudes_intervals_stats(qrs_filename)
      record_stats = np.append([record_index], [medians, 
                                                stds, 
                                                stds/medians]).reshape((1, 1 + 2 * 3))
      if record_index == 0:
        records_stats = record_stats
      else:
        records_stats = np.r_[records_stats, record_stats]
      record_index += 1
    return np.array(records).reshape((record_index, 1)), records_stats


def print_records_sorted_by_stats(records, sorted_stats, stats_filename, records_filename, thres, sortbycol = -2):
  with open(stats_filename, "w") as stats_file, \
       open(records_filename, "w") as records_file:
    print >> stats_file, "Record\tAmpMed\tIntMed\tAmpStd\tIntStd\tAmpStdToMed\tIntStdToMed"
    for ind in xrange(sorted_stats.shape[0]):
      print >> stats_file, records[int(sorted_stats[ind, 0]), 0] + "\t" + \
                           "\t".join(map(str, sorted_stats[ind, 1:]))
      if sorted_stats[ind, sortbycol] > thres:
        print >> records_file, records[int(sorted_stats[ind, 0]), 0]


def epanechnikov_kernel(x):
  return (1 - x ** 2)
  #return np.exp(-0.5 * x ** 2)
  
def get_kernel_noise_value_on_local_window(train_size, 
                                           valid_index, 
                                           window_values, 
                                           local_window):
  min_index = max(valid_index - local_window, 0)
  max_index = min(valid_index + local_window, train_size - 1)
  train_indices = np.append(np.arange(min_index, valid_index), 
                            np.arange(valid_index + 1, max_index + 1)) 
  weights = [epanechnikov_kernel((train_index - valid_index) * 1.0 / local_window) \
             for train_index in train_indices]
  
  weights_sum = sum(weights)
  window_value_estimate = sum([weight * window_values[train_index] \
                               for weight, train_index in zip(weights, train_indices)]) / weights_sum

  return abs(window_values[valid_index] - window_value_estimate)


def get_kernel_noise_variance_in_window(window_values, window_sizes):
  train_size = len(window_values)
  variances_by_window_sizes = []
  variances_by_window_sizes = [[get_kernel_noise_value_on_local_window(train_size,
                                                                       valid_index, 
                                                                       window_values,
                                                                       local_window_size + 1) \
                                                  for local_window_size in window_sizes] 
                                for valid_index in xrange(train_size)]

  variances_by_window_sizes = np.array(variances_by_window_sizes).reshape((train_size, len(window_sizes)))
  argmins = np.argmin(variances_by_window_sizes, axis = 1)
  mins = np.amin(variances_by_window_sizes, axis = 1)
  with open("argmins", "a") as outp:
    res = ""
    for window_argmin in argmins:
      res += str(window_sizes[window_argmin]) + "\t"
    res = res.strip()
    print >> outp, res

  with open("mins", "a") as outp:
    res = ""
    for err_min in mins:
      res += str(err_min) + "\t"
    res = res.strip()
    print >> outp, res
    
  meds = np.median(variances_by_window_sizes, axis = 0)
  stds = np.std(variances_by_window_sizes, axis = 0)

  return np.median(variances_by_window_sizes, axis = 0)



def get_noise_variance_in_window(window, window_sizes, dist_type = "EUCLID"):
  if dist_type == "KERNEL":
    variances = get_kernel_noise_variance_in_window(window, window_sizes)
    return variances

  window_size = window_sizes
  slope = (window[-1] - window[0]) / (window_size - 1)
  free_term = window[0]
  variance = 0
  for i in xrange(1, window_size - 1):
      if dist_type == "EUCLID":
        orthog_free_term = window[i] + slope * i
        if abs(slope) > 0: 
          intersect_i = (free_term - orthog_free_term) / (2 * slope)
          distance = (window[i] - slope * intersect_i - free_term) ** 2 + \
                     (i - intersect_i) ** 2
        else:
          distance = window[i] ** 2
        variance += distance
      elif dist_type == "HORIZ":
        variance += (window[i] - slope * i - free_term) ** 2
  return (variance / window_size) ** 0.5


def estimate_record_noise(patient_record, 
                          window_size, 
                          signal = 0, 
                          dist_type = "EUCLID"):
  window_values = []
  variance = 0
  variances = []
  for line in open(patient_record):
      signals = re.split("\s+", line.strip())[1:]
      signal_value = float(signals[signal])
      
      window_values.append(signal_value)
      if dist_type != "KERNEL":
        if len(window_values) >= 1.5 * window_size:
          window_values = window_values[-window_size:]
        if len(window_values) == window_size:
          variance = get_noise_variance_in_window(window_values, window_size, dist_type)
          variances.append(variance)
  if dist_type != "KERNEL":
    med = np.median(variances)
    std = np.std(variances)
    variances = np.array(variances)
    filtered_variances = variances[variances <= med + 3 * std]
    return np.median(filtered_variances)
  else:
    return get_noise_variance_in_window(window_values, window_size, dist_type)


def estimate_diagnosis_noise(patients_filename, 
                             signal = 0, 
                             dist_type = "EUCLID", 
                             window_sizes = []):
  variances = []
  """
  f = open("argmins", "w")
  f.close()
  f = open("mins", "w")
  f.close()
  """

  for patient in open(patients_filename):
    patient_record_filename = "ptbdb/" + patient.strip()
    with open("argmins", "a") as outp:
      print >> outp, patient.strip()
    with open("mins", "a") as outp:
      print >> outp, patient.strip()
    variance = estimate_record_noise(patient_record_filename, window_sizes, signal, dist_type)
    variances.append(variance)
    print len(variances)
    print patient
  return np.array(variances)


def print_stats_and_noise(output_filename, records, stats, noise_vars):
  with open(output_filename, "w") as output:
    for i in xrange(records.shape[0]):
      print >> output, records[i, 0] + \
                       "\t" + "\t".join(map(str, stats[i, 1:])) + \
                       "\t" + "\t".join(map(str, noise_vars[i, :]))


def plot_noise_by_window_size(window_sizes, noise_filename, path, dist_type = "EUCLID"):
  for line in open(noise_filename):
    line = line.strip().split("\t")
    patient = line[0]
    noises = map(float, line[-len(window_sizes):])
    fig = plt.plot(noises)
  
  plt.xticks(range(len(window_sizes)), window_sizes)
  plt.savefig(path + dist_type.lower() + "_dist" + ".png")

  
def estimate_diagnosis_noise_by_window_sizes(patients_filename, window_sizes, signal = 0, dist_type = "EUCLID"):
  print window_sizes
  if dist_type != "KERNEL":
    for i in xrange(len(window_sizes)):
      print window_sizes[i]
      if i == 0:
        variances = estimate_diagnosis_noise(patients_filename_, signal, dist_type, window_sizes[i]) 
      else:
        variances = np.c_[variances, estimate_diagnosis_noise(patients_filename_, signal, dist_type, window_sizes[i])]
      print variances.shape
  else:
    variances = estimate_diagnosis_noise(patients_filename_, signal, dist_type, window_sizes)
  return variances

from collections import defaultdict
def get_most_frequent_window_size():
  with open("argmins") as inp, open("most_frequent_window_sizes", "w") as outp:
    for line in inp:
      line = line.strip().split("\t")
      if len(line) == 1:  
        distr = defaultdict(int)
        print >> outp, line[0]
      else:
        for window_size in line:
          distr[window_size] += 1
        sorted_distrs = sorted([(key, value) for key, value in distr.iteritems()], 
                               key = lambda x: x[1], reverse = True)
        print >> outp, "\t".join(map(lambda x: "(" + str(x[0]) + ", " + str(x[1]) + ")", sorted_distrs))

def get_noise_distribution_description():
  with open("mins") as inp, open("noise_distribution_description", "w") as outp:
    for line in inp:
      line = line.strip().split("\t")
      if len(line) == 1:  
        distr = defaultdict(int)
        print >> outp, line[0]
      else:
        line = np.array(map(float, line))
        noise_med = np.median(line)
        noise_mean = np.mean(line)
        noise_std = np.std(line)
        filtered_line = line[line < noise_mean + 2 * noise_std]
        filtered_noise_med = np.median(filtered_line)
        print np.amax(line), np.mean(line)
        print >> outp, "(Med %.4f, Std %.4f, FilMed %.4f)" % (noise_med, noise_std, filtered_noise_med)

    
diagnosis = "healthy_control"
if len(sys.argv) > 1:
  diagnosis = sys.argv[1]
window_size = 100
if len(sys.argv) > 2:
  window_size = int(sys.argv[2])
thres = 0.1
if len(sys.argv) > 3:
  thres = float(sys.argv[3])
signal = 0
if len(sys.argv) > 4:
  signal = int(sys.argv[4])

diagnosis_path = "/Users/shushu/Documents/WFDB/diagnosis/" + diagnosis + "_stats/" 
patients_filename_ = "/Users/shushu/Documents/WFDB/diagnosis/" + diagnosis + "_subset"
#patients_filename_ = diagnosis_path + "noisy"
stats_filename_ = diagnosis_path + "stats" 
noise_with_stats_filename = diagnosis_path + "stats_noise" 
#sorted_by_amp_filename_ = diagnosis_path + "recs_stats_sorted_by_amp" 
#sorted_by_int_filename_ = diagnosis_path + "recs_stats_sorted_by_int"
#recs_sorted_by_amp_filename_ = diagnosis_path + "noisy_by_amp" 
#recs_sorted_by_int_filename_ = diagnosis_path + "noisy_by_int"

records, stats = get_records_stats(patients_filename_)
print_stats(stats_filename_, records, stats)

window_sizes = []
cnt_min =  0
cnt_max = 3
for i in xrange(cnt_min, cnt_max):
  window_sizes.append(1000 + i * 400)
window_sizes = sorted(set(window_sizes))

approx_type = "KERNEL"
window_sizes = sorted(set([int(np.exp(i * 0.3)) for i in xrange(5, 22)]))
window_sizes = [2 * i - 1 for i in xrange(1, 20)]
#window_sizes = sorted(set([int(np.exp(i * 0.3)) for i in xrange(2, 7)]))
 
#noise_variances = estimate_diagnosis_noise_by_window_sizes(patients_filename_, window_sizes, signal, approx_type)
#print_stats_and_noise(noise_with_stats_filename, records, stats, noise_variances)
#plot_noise_by_window_size(window_sizes, noise_with_stats_filename, diagnosis_path, approx_type)
get_most_frequent_window_size()
get_noise_distribution_description()  
