 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import re
import sys
import numpy as np

import os
os.chdir("/Users/shushu/Documents/WFDB/")


def split_line(line):
  return re.split("\s+", line.strip())


def load_qrs_occurencies(filename):
  qrs_occurencies = []
  with open(filename) as f:
    for line in f:
      qrs_occurencies.append(int(split_line(line)[1]))
  if qrs_occurencies[0] < 0:
    qrs_occurencies[0] = 0
  return qrs_occurencies


def print_qrs_amplitudes_intervals(r_samples, amplitudes, intervals, qrs_output_filename):
  with open(qrs_output_filename, "w") as qrs_output_file:
    print >> qrs_output_file, "Sample\tAmplitude\tInterval\tToPrevRSample"
    prev_r_sample = 0
    omitted_intervals = 0
    for r_sample, amplitude, interval in zip(r_samples, amplitudes, intervals):
      is_omitted = ""
      if r_sample - prev_r_sample > interval:
        omitted_intervals += 1
        is_omitted = "O"
      print >> qrs_output_file, "%d\t%d\t%d\t%d\t%s" % (r_sample, amplitude, interval, r_sample -
          prev_r_sample, is_omitted)
      prev_r_sample = r_sample
    return omitted_intervals, len(r_samples)


def get_robust_median_and_stdev(array_values, std_is_modified = True):
  median = np.median(array_values)
  if std_is_modified:
    stdev = np.sqrt(np.median(abs(array_values - median) ** 2)) 
  else:
    stdev = np.sqrt(np.mean(abs(array_values - median) ** 2))
  return median, stdev 


def filter_array(array_values, by_max_thres = True, by_min_thres = True, nsigma = 4):
  median, stdev = get_robust_median_and_stdev(array_values)
  filtered_values = array_values 
  if not by_min_thres:
    nsigma += 1
  if by_max_thres:
    filtered_values = filtered_values[filtered_values <= median + nsigma * (stdev + 10)] 
  else:
    nsigma += 1
  if by_min_thres:
    filtered_values = filtered_values[filtered_values >= median - nsigma * (stdev + 10)]
  return filtered_values


def get_confidence_interval(values, n = 3, stdev_add_term = 20, filter_by_max = True, filter_by_min = True):
  array_values = np.array(values)
  filtered_values = filter_array(array_values, filter_by_max, filter_by_min, nsigma = n + 1)
  median, stdev = get_robust_median_and_stdev(filtered_values, std_is_modified = False)
  stdev += stdev_add_term
  return int(median - n * stdev), int(median + n * stdev)


def edit_qrs_qrs_amplitudes_intervals(r_samples, amplitudes, intervals):
  if len(r_samples) == 0:
    return r_samples, amplitudes, intervals, 0, 0, 0, 0
  amplitude_min_value, amplitude_max_value = get_confidence_interval(amplitudes, n = 4, stdev_add_term = 150)
  amplitude_min_value, amplitude_max_value = max(amplitude_min_value, 300), min(amplitude_max_value, 5700)
  if amplitude_max_value < amplitude_min_value:
    amplitude_max_value += amplitude_min_value
  interval_min_value, interval_max_value = get_confidence_interval(intervals, n = 4, stdev_add_term = 100, filter_by_min = False)
  interval_min_value, interval_max_value = max(interval_min_value, 300), min(interval_max_value, 2000)
  if interval_min_value > interval_max_value:
    interval_max_value += interval_min_value

  united_interval = 0
  united_intervals_count = 0
  corrected_r_samples = [r_samples[0],]
  corrected_amplitudes = [amplitudes[0],]
  corrected_intervals = [intervals[0],]
  for r_sample, amplitude, interval in zip(r_samples[1:], amplitudes[1:], intervals[1:]):
    new_united_interval = interval + united_interval
    united_intervals_count += 1
    if new_united_interval <= interval_max_value:
      united_interval = new_united_interval
    elif united_interval > 0 and interval <= interval_max_value:
      united_interval = interval
      united_intervals_count = 1
    elif new_united_interval <= united_intervals_count * interval_max_value:
      united_interval = new_united_interval / united_intervals_count
      united_intervals_count = 1
     
    if amplitude_min_value <= amplitude <= amplitude_max_value:
      if interval_min_value <= united_interval <= interval_max_value:
        corrected_intervals.append(united_interval)
        corrected_r_samples.append(r_sample)
        corrected_amplitudes.append(amplitude)
        united_interval = 0
        united_intervals_count = 0
  return corrected_r_samples, corrected_amplitudes, corrected_intervals, amplitude_min_value, amplitude_max_value, \
         interval_min_value, interval_max_value 


def get_qr_intervals_thres(qr_intervals,
                           new_value, 
                           nsigma, 
                           max_thres_max_value,
                           min_thres_min_value, 
                           stdev_add_term = 20, 
                           filter_by_max = True, 
                           filter_by_min = True):
  qr_intervals = np.append(qr_intervals, new_value)
  if len(qr_intervals) > 5:
    qr_interval_min_thres, qr_interval_max_thres = get_confidence_interval(qr_intervals, 
                                                                           n = nsigma, 
                                                                           stdev_add_term = stdev_add_term,
                                                                           filter_by_max = filter_by_max, 
                                                                           filter_by_min = filter_by_min)
    if filter_by_max:
      qr_interval_max_thres = min(qr_interval_max_thres, max_thres_max_value)
    if filter_by_min:
      qr_interval_min_thres = max(qr_interval_min_thres, min_thres_min_value)
    if qr_interval_max_thres < qr_interval_min_thres:
      qr_interval_min_thres = qr_interval_max_thres - 20
      qr_interval_max_thres += 10
  else:
    qr_interval_min_thres = min_thres_min_value
    qr_interval_max_thres = max_thres_max_value
  return qr_intervals, qr_interval_min_thres, qr_interval_max_thres 


def window_is_not_noisy(window, max_thres):
  return bool(max(window) - min(window) < max_thres)


def find_min_or_max_of_window(window, index_of_last_window_element, min_or_max = "MAX"):
  window_values = np.array(window)
  if min_or_max == "MIN":
    window_values = -1 * window_values
  window_max_index = np.argmax(window_values)
  return window_max_index, window[window_max_index], index_of_last_window_element - (len(window) - window_max_index - 1)


def get_qrs_amplitudes_intervals(data_filename, q_samples, signal = 0):
  r_samples = []
  amplitudes = []
  intervals = []
  with open(data_filename) as data_file, open("f", "w") as outp:
    q_samples_count = len(q_samples)
    current_q = 0
    last_r_sample = 0
    qr_intervals = np.empty([0])
    qr_amplitudes = np.empty([0])
    q_size = Q_SIZE = 20
    S_SIZE = 100
    pq_size = 30
    q_window = []
    QRS_SIZE = 100
    MINUS_INF = -1e10    
    min_of_q_s_value = min_of_q_values = 0
    qr_interval_min_value, qr_interval_max_value = 20, 120
    qr_amplitude_min_value, qr_amplitude_max_value = 300, 5500
    qr_interval_min_thres, qr_interval_max_thres = qr_interval_min_value, qr_interval_max_value
    qr_amplitude_min_thres, qr_amplitude_max_thres = qr_amplitude_min_value, qr_amplitude_max_value 

    
    for line in data_file:
      splitted_line = split_line(line)
      sample = int(splitted_line[0])
      signal_value = int(splitted_line[1 + signal])
      if current_q < q_samples_count and 0 <= q_samples[current_q] - sample < pq_size:
        q_window.append(signal_value)
      #если встретили Q зубец
      if current_q < q_samples_count and sample == q_samples[current_q] or \
        current_q == q_samples_count and \
        sample >= q_samples[current_q - 1] + Q_SIZE + QRS_SIZE + S_SIZE:
        if current_q > 0:
          r_samples.append(r_sample)
          amplitudes.append(r_value - min_of_q_s_value)
          intervals.append(r_sample - last_r_sample)
          print >> outp, "\nQCURR= ", current_q, "QSIZE= ", q_size
          print >> outp, "QRAmpMin= %.2f\tQRAmpMax= %.2f" %( qr_amplitude_min_thres, qr_amplitude_max_thres)
          print >> outp, "R= ", r_value
          print >> outp, "RSamp= %d\tQorSMin= %d\tQMin= %d\tCorrQ= %d" % (r_sample, min_of_q_s_value,
              min_of_q_values, corrected_q_sample)
          print >> outp, "QRS_SIZE= %.2f\tQRIntMinThres= %.2f,\tQRIntMinThres= %.2f" % (QRS_SIZE,
              qr_interval_min_thres,
              qr_interval_max_thres)

          if r_sample > corrected_q_sample:
            #print qr_intervals
            qr_intervals, qr_interval_min_thres, qr_interval_max_thres = \
                  get_qr_intervals_thres(qr_intervals,
                                         r_sample - corrected_q_sample, 
                                         nsigma = 4, 
                                         max_thres_max_value = qr_interval_max_value,
                                         min_thres_min_value = qr_interval_min_value,
                                         stdev_add_term = 10)

            qr_amplitudes, qr_amplitude_min_thres, qr_amplitude_max_thres = \
                  get_qr_intervals_thres(qr_amplitudes, 
                                         r_value - min_of_q_values, 
                                         nsigma = 4, 
                                         #filter_by_max = False,
                                         max_thres_max_value = qr_amplitude_max_value,
                                         min_thres_min_value = qr_amplitude_min_value, 
                                         stdev_add_term = 100)
            
          last_r_sample = r_sample
        r_value, r_sample = MINUS_INF, sample
        min_of_q_s_value = min_of_q_values = signal_value
        half_window_size = min(qr_interval_min_thres / 2, 6) 
        window_size = 2 * half_window_size + 1 
        window = []
        QRS_SIZE = 2 * qr_interval_max_thres
        q_size = Q_SIZE
        corrected_q_sample = q_sample = sample 
        q_shift = 0
        if len(q_window) > 0 and window_is_not_noisy(q_window, qr_amplitude_max_thres):
          window_min_index, window_min_value, window_min_sample = find_min_or_max_of_window(q_window, sample, min_or_max = "MIN")
          if window_min_value < signal_value:
            corrected_q_sample = q_sample = window_min_sample
            min_of_q_s_value = min_of_q_values = window_min_value
            q_shift = sample - q_sample
            q_size += q_shift
            window = q_window[window_min_index:]
        last_r_peak = corrected_q_sample
        r_value, r_sample = MINUS_INF, last_r_peak

        q_window = []
        current_q += 1

      elif current_q > 0:
        window.append(signal_value)
        if window_size - 1 <= sample - q_sample:
          window = window[-window_size:]
          window_middle_sample = sample - half_window_size - 1
          window_middle_value = window[-half_window_size - 1]
          window_not_noisy = window_is_not_noisy(window, qr_amplitude_max_thres)
          if sample - q_sample < q_size:
            if q_size < q_shift + Q_SIZE + 20: 
              if window_not_noisy:
                if window_middle_value < min_of_q_s_value:
                  min_of_q_s_value = min_of_q_values = window_middle_value
                  corrected_q_sample = window_middle_sample 
                  if corrected_q_sample - q_sample > q_size / 2:
                    q_size += 5
                  last_r_peak = corrected_q_sample
                  r_sample, r_value = window_middle_sample, window_middle_value
              else:
                q_size += 20 / float(window_size)
          if window_middle_sample > corrected_q_sample and window_not_noisy:
            #если встретили S зубец
            if 0 < window_middle_sample - r_sample < qr_interval_max_thres + S_SIZE:
              min_of_q_s_value = min(min_of_q_s_value, window_middle_value)
            #если встретили R зубец
            if window_middle_value > r_value and \
               max(qr_amplitude_min_thres, qr_amplitude_min_value) < window_middle_value - min_of_q_values < \
               qr_amplitude_max_thres:
              if last_r_peak == corrected_q_sample or window_middle_sample - last_r_peak < qr_interval_max_thres:
                r_sample, r_value = window_middle_sample, window_middle_value
                last_r_peak = r_sample
  return r_samples, amplitudes, intervals             

log_filename=sys.argv[1]
patient = sys.argv[2]
local_path = "ptbdb/"

hd_path = local_path
if len(sys.argv) > 3:
  hd_path = sys.argv[3] + "/"
data_filename = hd_path + patient
qrs_input_filename = hd_path + patient + "_sqrs_output"

qrs_output_filename = local_path + patient + "_qrs"
corrected_qrs_output_filename = qrs_output_filename + "_corrected"

q_samples = load_qrs_occurencies(qrs_input_filename)
r_samples_, amplitudes_, intervals_ = get_qrs_amplitudes_intervals(data_filename, q_samples)
print_qrs_amplitudes_intervals(r_samples_, amplitudes_, intervals_, qrs_output_filename)
r_samples_, amplitudes_, intervals_, amplitude_min, amplitudes_max, interval_min, interval_max = \
                                      edit_qrs_qrs_amplitudes_intervals(r_samples_, 
                                                                        amplitudes_,
                                                                        intervals_) 
omitted_intervals, total = print_qrs_amplitudes_intervals(r_samples_, amplitudes_, intervals_, corrected_qrs_output_filename)
with open(log_filename, "a") as outp:
  print >> outp, "Patient=%s\tTotal=%d\tOmittedIntervals=%d\tAmpMin=%d\tAmpMax=%d\tIntMin=%d\tIntMax=%d" % (patient, 
                                                                                                            total,
                                                                                                            omitted_intervals, 
                                                                                                            amplitude_min, 
                                                                                                            amplitudes_max, 
                                                                                                            interval_min, 
                                                                                                            interval_max)
print "Omitted=%d\tTotalCorrected=%d\tTotalInput=%d" % (omitted_intervals, total, len(q_samples))
#print amplitude_min, amplitudes_max, interval_min, interval_max
