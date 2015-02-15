 #!/usr/bin/python
 # -*- coding: utf-8 -*-

import re
import sys
import numpy as np

import os
os.chdir("/Users/shushu/Documents/WFDB/")


MAX_AMP=5600
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


def print_qrs_amplitudes_intervals(r_peaks, qrs_output_filename):
  with open(qrs_output_filename, "w") as qrs_output_file:
    print >> qrs_output_file, "Sample\tAmplitude\tInterval\tArgminOfQS\tQSample\tTSample\tIsOmitted"
    prev_r_sample = 0
    omitted_intervals = 0
    for r_peak in r_peaks:
      is_omitted = ""
      if r_peak.r_sample - prev_r_sample > r_peak.interval:
        omitted_intervals += 1
        is_omitted = "O"
        r_peak.interval = r_peak.r_sample - prev_r_sample
      print >> qrs_output_file, "%s\t%s" % (r_peak, is_omitted)
      prev_r_sample = r_peak.r_sample
    return omitted_intervals, len(r_peaks)


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


class RPeak(object): 
  def __init__(self, r_sample, amplitude, interval, argmin_of_qs_samples, q_sample, t_sample):
    self.r_sample, self.amplitude, self.interval, self.argmin_of_qs_samples, self.q_sample, self.t_sample = \
      r_sample, amplitude, interval, argmin_of_qs_samples, q_sample, t_sample

  def __str__(self):
    return "\t".join(map(str, [self.r_sample, self.amplitude, self.interval, \
                               self.argmin_of_qs_samples, self.q_sample, self.t_sample]))


def edit_qrs_qrs_amplitudes_intervals(r_peaks):
  r_samples = [x.r_sample for x in r_peaks]
  amplitudes = [x.amplitude for x in r_peaks]
  intervals = [x.interval for x in r_peaks]

  if len(r_samples) == 0:
    return r_samples, amplitudes, intervals, 0, 0, 0, 0
  amplitude_min_value, amplitude_max_value = get_confidence_interval(amplitudes, n = 4, stdev_add_term = 150)
  amplitude_min_value, amplitude_max_value = max(amplitude_min_value, 300), min(amplitude_max_value, MAX_AMP + 200)
  if amplitude_max_value < amplitude_min_value:
    amplitude_max_value += amplitude_min_value
  interval_min_value, interval_max_value = get_confidence_interval(intervals, n = 4, stdev_add_term = 100, filter_by_min = False)
  interval_min_value, interval_max_value = max(interval_min_value, 300), min(interval_max_value, 2000)
  if interval_min_value > interval_max_value:
    interval_max_value += interval_min_value

  united_interval = 0
  united_intervals_count = 0
  corrected_r_peaks = [r_peaks[0],]
  for r_peak in r_peaks[1:]:
    new_united_interval = r_peak.interval + united_interval
    united_intervals_count += 1
    if new_united_interval <= interval_max_value:
      united_interval = new_united_interval
    elif united_interval > 0 and r_peak.interval <= interval_max_value:
      united_interval = r_peak.interval
      united_intervals_count = 1
    elif new_united_interval <= united_intervals_count * interval_max_value:
      united_interval = new_united_interval / united_intervals_count
      united_intervals_count = 1
     
    if amplitude_min_value <= r_peak.amplitude <= amplitude_max_value:
      if interval_min_value <= united_interval <= interval_max_value:
        corrected_r_peaks.append(r_peak)
        united_interval = 0
        united_intervals_count = 0
  return corrected_r_peaks, amplitude_min_value, amplitude_max_value, interval_min_value, interval_max_value 


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


def get_qrs_amplitudes_intervals(data_filename, data_line_count, q_samples, signal = 0):
  r_peaks = []
  if data_line_count == 0:
    return r_peaks

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
    qr_amplitude_min_value, qr_amplitude_max_value = 300, MAX_AMP
    qr_interval_min_thres, qr_interval_max_thres = qr_interval_min_value, qr_interval_max_value
    qr_amplitude_min_thres, qr_amplitude_max_thres = qr_amplitude_min_value, qr_amplitude_max_value 

    
    for line in data_file:
      splitted_line = split_line(line)
      if len(splitted_line) > 1 + signal:
        sample = int(splitted_line[0])
        signal_value = int(splitted_line[1 + signal])
        if current_q < q_samples_count and 0 <= q_samples[current_q] - sample < pq_size:
          q_window.append(signal_value)
        #если встретили Q зубец
        if current_q < q_samples_count and sample == q_samples[current_q] or \
          current_q == q_samples_count and \
          sample == min(q_samples[current_q - 1] + Q_SIZE + QRS_SIZE + S_SIZE, data_line_count - 1):
          if current_q > 0:
            r_peaks.append(RPeak(r_sample, r_value - min_of_q_s_value, \
                                 r_sample - last_r_sample, argmin_of_q_s_sample, \
                                 q_sample, t_sample))
            print >> outp, "\nQCURR= ", current_q, "QSIZE= ", q_size
            print >> outp, "QRAmpMin= %.2f\tQRAmpMax= %.2f" % (qr_amplitude_min_thres, qr_amplitude_max_thres)
            print >> outp, "R= ", r_value
            print >> outp, "RSamp= %d\tCorrQ= %d\tSSamp= %d\tTSamp= %d\t" % (r_sample, 
                                                                             corrected_q_sample, 
                                                                             s_sample, 
                                                                             t_sample)
            print >> outp, "QRS_SIZE= %.2f\tQRIntMinThres= %.2f,\tQRIntMaxThres= %.2f" % (QRS_SIZE,
                                                                                          qr_interval_min_thres,
                                                                                          qr_interval_max_thres)
            #if r_sample < q_sample or s_sample < r_sample or t_sample < s_sample:
            #  print "WOHOO", current_q

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
                                           #filter_by_min = False,
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
          q_size = Q_SIZE + 60
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
          s_sample, s_value = t_sample, t_value = r_sample, r_value
          argmin_of_q_s_sample = corrected_q_sample  
          q_window = []
          current_q += 1

        elif 0 < current_q <= q_samples_count:
          window.append(signal_value)
          if window_size - 1 <= sample - q_sample:
            window = window[-window_size:]
            window_middle_sample = sample - half_window_size - 1
            #window_middle_sample = sample - half_window_size 
            window_middle_value = window[-half_window_size - 1]
            window_not_noisy = window_is_not_noisy(window, qr_amplitude_max_thres)
            if sample - q_sample < q_size:
              if q_size < q_shift + Q_SIZE + 50: 
                if window_not_noisy:
                  if window_middle_value < min_of_q_values:
                    min_of_q_s_value = min_of_q_values = window_middle_value
                    argmin_of_q_s_sample = corrected_q_sample = window_middle_sample 
                    if corrected_q_sample - q_sample > q_size / 4:
                      q_size += 10
                    last_r_peak = corrected_q_sample
                    r_sample, r_value = window_middle_sample, window_middle_value
                else:
                  q_size += 20 / float(window_size)
            if window_middle_sample > corrected_q_sample and window_not_noisy:
              #если встретили S зубец
              if 0 < window_middle_sample - r_sample < qr_interval_max_thres + S_SIZE:
                if window_middle_value < s_value: 
                  s_sample, s_value = window_middle_sample, window_middle_value
                  t_sample, t_value = window_middle_sample, window_middle_value
                
                if window_middle_value < min_of_q_s_value:
                  argmin_of_q_s_sample, min_of_q_s_value = window_middle_sample, window_middle_value 
              
              T_SIZE = qr_interval_max_thres
              #если встретили T зубец
              if 0 < window_middle_sample - s_sample < T_SIZE:
                if window_middle_value > t_value:
                  t_sample, t_value = window_middle_sample, window_middle_value
            
                  
              #если встретили R зубец
              if window_middle_value > r_value and \
                 max(qr_amplitude_min_thres, qr_amplitude_min_value) < window_middle_value - min_of_q_values < \
                 qr_amplitude_max_thres:
                if last_r_peak == corrected_q_sample or window_middle_sample - last_r_peak < qr_interval_max_thres:
                  r_sample, r_value = window_middle_sample, window_middle_value
                  last_r_peak = r_sample
                  s_sample, s_value = r_sample, r_value
  return r_peaks 


def next_q_sample_is_reached(sample, r_peaks, r_peak_index):
  return r_peak_index < len(r_peaks) and sample == r_peaks[r_peak_index].q_sample


def is_between_prev_t_and_next_q_samples(sample, r_peaks, prev_t_sample, r_peak_index, data_line_count):
  return sample + 1 < data_line_count and \
         (r_peak_index == len(r_peaks) or sample < r_peaks[r_peak_index].q_sample) and \
         sample >= prev_t_sample


def window_is_noisy_centered(window, amplitudes_max, window_size):
  if max(window) - min(window) > amplitudes_max:
    return min(abs(window[window_size / 2] - window[0]), \
               abs(window[window_size / 2] - window[-1])) > amplitudes_max
  return False
    
def get_noisy_segments(data_filename, data_line_count, r_peaks, amplitudes_max, signal = 0):
  window = []
  segments = []
  segment = []
  window_size = 30
  min_segment_size = 80
  r_peak_index = 0
  prev_t_sample = 0
  for line in open(data_filename):
    line = split_line(line)
    sample = int(line[0])
    signal_value = int(line[1 + signal])
    if is_between_prev_t_and_next_q_samples(sample, r_peaks, prev_t_sample, r_peak_index, data_line_count): 
      segment.append([sample, signal_value])
      window.append(signal_value)
      if len(window) > window_size:
        window = window[-window_size:]
        window_is_noisy = window_is_noisy_centered(window, amplitudes_max, window_size)
        if window_is_noisy:
          window = []
          segment = segment[:-window_size]
          if len(segment) > min_segment_size:
            segments.append(segment[:-window_size])
          segment = []
    else: 
      if next_q_sample_is_reached(sample, r_peaks, r_peak_index):
        window = []
        if len(segment) > min_segment_size:
          segments.append(segment)
        segment = []

        prev_t_sample = r_peaks[r_peak_index].t_sample
        r_peak_index += 1
  return segments


def get_window_and_matrix(segment, matrix, first_index, last_index, index, window_size, train_indices):
  """
  window = segment[first_index : last_index]
  window = np.delete(window, (index - first_index), axis = 0)

  matrix = matrix[: window_size + 1, -window_size :]
  matrix = np.delete(matrix, (index - first_index), axis = 0)
  """
  window = segment[train_indices]
  matrix = matrix[train_indices - first_index, -window_size : ]
  return window, matrix 


def get_first_last_indices(index, window_size, max_index):
  window_size = 2 * int(window_size * 0.5)
  first_index = max(0, index - window_size / 2)
  last_index = min(index + window_size / 2 + 1, max_index)
  if last_index - first_index < window_size + 1:
    first_index = max(0, last_index - window_size - 1)
  window_size = last_index - first_index - 1
  return first_index, last_index, window_size


from matplotlib import pyplot as plt
def plot_approximation(first_index, last_index, segment, curr_window_size, mult, coeffs):
  xvalues = np.arange(first_index, last_index)
  plt.plot(xvalues, segment[first_index : last_index])
  
  xvalues_ = np.arange(0, curr_window_size + 1, 0.1) - curr_window_size / 2
  matrix = np.vander(xvalues_ * mult, curr_window_size)
  plt.plot(np.arange(first_index, last_index, 0.1), \
           [np.dot(coeffs, matrix[ind, -curr_window_size:]) for ind in xrange(matrix.shape[0])])
  plt.ylim(-500, -300)
  plt.savefig("FFFFF")
  

def get_test_indices(window_size, first_index, index, last_index):
    min_left = first_index + 6
    max_right = last_index - 6
    if window_size > 20 and index > min_left and index < max_right:
      test_indices = np.arange(max(first_index + 6, index - window_size / 4), min(index + window_size / 4, last_index - 6), 3)
    else:
      test_indices = np.array([index, ])
    train_indices = np.sort(list(set(np.arange(first_index, last_index)) - set(test_indices)))
    window_size = train_indices.shape[0]

    return test_indices, train_indices, window_size

      
def estimate_each_segment_pointwise_noise(segment, window_size, matrix, mult, step = 1):
  segment_noise = []
  segment_size = len(segment)
  for index in xrange(0, segment_size, step):
     
    first_index, last_index, curr_window_size = get_first_last_indices(index, window_size, segment_size)
    test_indices, train_indices, curr_window_size = get_test_indices(curr_window_size, first_index, index, last_index)
    window, window_matrix = get_window_and_matrix(segment, matrix, first_index, last_index, index, curr_window_size, \
                                                  train_indices)
    coeffs = np.linalg.solve(window_matrix, window)
    estimates = np.dot(coeffs, matrix[test_indices - first_index, -curr_window_size:].T)
    if len(test_indices) == 1:
      segment_noise.append(abs(estimates - segment[index]))
    else:  
      filtered = filter_array(estimates - segment[test_indices], by_max_thres = True, by_min_thres = True, nsigma = 5)
      if len(filtered) > 4:
        segment_noise.append(np.std(filtered))
      else:
        segment_noise.append(max(abs(filtered)))
      #segment_noise.append(get_robust_median_and_stdev(estimates - segment[test_indices], std_is_modified = True)[1])
    """
    estimate = np.dot(coeffs, matrix[index - first_index, -curr_window_size:])
    segment_noise.append(abs(estimate - segment[index]))
    """

    """
    if index == 500:
      plot_approximation(first_index, last_index, segment, curr_window_size, mult, coeffs)
    """
  return segment_noise


def estimate_segments_pointwise_noise(segments, window_size, mult = 0.001):
  segment_noises = []
  #segments = [segments[0], ]
  step = 5
  matrix = np.vander((np.arange(window_size + 1) - window_size / 2) * mult, window_size)
  for segment in segments:
    segment_size = len(segment)
    segment_values = np.array(segment)[:, 1]
    segment_window_size = min(window_size, segment_size)
    segment_noises.append(estimate_each_segment_pointwise_noise(segment_values, segment_window_size, matrix, mult, step))
  return segment_noises


def estimate_segments_overall_noise(segments, segments_pointwise_noises):
  segments_overall_noises = []
  for segment_noise in segments_pointwise_noises:
    filtered = filter_array(np.array(segment_noise), by_max_thres = True, by_min_thres = True, nsigma = 5)
    segments_overall_noises.append(np.median(filtered))
    #segments_overall_noises.append(np.sqrt(np.mean(filtered ** 2)))
  return segments_overall_noises   


def estimate_pointwise_and_record_noise(segments, window_size = 20, mult = 0.001):
  segments_pointwise_noises = estimate_segments_pointwise_noise(segments, window_size, mult)
  segments_overall_noises = estimate_segments_overall_noise(segments, segments_pointwise_noises)
  record_noise = np.median(segments_overall_noises)
  return segments_pointwise_noises, record_noise


def print_pointwise_noise(segment_noises_filename, segments, segments_pointwise_noises):
  with open(segment_noises_filename, "w") as out:
    for segment, segment_noise in zip(segments, segments_pointwise_noises):
      for ind in xrange(len(segment)):
        print >> out, "%d\t%d\t%.4f" % (segment[ind][0], segment[ind][1], segment_noise[ind])
  with open(patients_noises_filename, "a") as out2:
    print >> out2, "%s\t%.4f" % (patient, record_noise)


class NoiseByWindow(object):
  def __init__(self, window_size, noise_value, mult):
    self.window_size = window_size
    self.noise_value = noise_value
    self.mult = mult

  def __str__(self):
    return str(self.noise_value) + ";" + str(self.mult)
    

def get_noises_by_window_size(segments): 
  noises_by_window_size = []
  for window_size in xrange(25, 50, 3):
    print window_size
    mult_argmin = 1
    min_noise = 1e10
    for mult in xrange(2, 4):
      print mult
      mult = 10 ** (-mult)
      _, record_noise = estimate_pointwise_and_record_noise(segments, window_size, mult)
      if record_noise <= min_noise:
        mult_argmin = mult
        min_noise = record_noise
    noises_by_window_size.append(NoiseByWindow(window_size, min_noise, mult_argmin))
  return noises_by_window_size
  
  
def print_noises_by_window_size(output_filename, patient, noises_by_window_size):
  window_sizes = map(lambda x: str(x.window_size), noises_by_window_size)
  with open(output_filename, "a") as out:
    print >> out, "%s\t%s" % (patient, "\t".join(map(str, noises_by_window_size)))


def read_r_peaks(data_filename):
  r_peaks = []
  with open(data_filename) as data:
    data.readline()
    for line in data:
      line = line.strip().split("\t")
      try:
        line = map(int, line[:6])
      except ValueError:
        line = [int(line[0]), float(line[1])] + map(int, line[2 : 6])
      r_peaks.append(RPeak(*line))
  return r_peaks


def process_noise(r_peaks_filename, data_filename, lines_count, patient, 
                  noise_by_window_size_filename, 
                  noise_filename, 
                  signal = 0, 
                  mode = "ONE"
                  ):
  r_peaks = read_r_peaks(r_peaks_filename)
  amp_med, amp_std = get_robust_median_and_stdev(map(lambda x: x.amplitude, r_peaks), std_is_modified = False)
  segments = get_noisy_segments(data_filename, lines_count, r_peaks, amp_med + 3 * amp_std, signal = 0)
  if mode == "ONE":
    window_size = 30
    mult = 0.001
    record_noise = estimate_pointwise_and_record_noise(segments, window_size, mult)[1]
    with open(noise_filename, "w") as out:
      print >> out, "%.4f\t%d" % (record_noise, len(r_peaks))
  else:
    noises_by_window_size = get_noises_by_window_size(segments)
    print_noises_by_window_size(noise_by_window_size_filename, patient, noises_by_window_size)


def calculate_r_peaks(data_filename, lines_count, qrs_input_filename, \
                      qrs_output_filename, corrected_qrs_output_filename, \
                      patient, log_filename, signal = 0):
  q_samples = load_qrs_occurencies(qrs_input_filename)
  r_peaks = get_qrs_amplitudes_intervals(data_filename, lines_count, q_samples, signal = signal)
  print_qrs_amplitudes_intervals(r_peaks, qrs_output_filename)

  r_peaks, amplitude_min, amplitudes_max, interval_min, interval_max = edit_qrs_qrs_amplitudes_intervals(r_peaks) 
  omitted_intervals, total = print_qrs_amplitudes_intervals(r_peaks, corrected_qrs_output_filename)
  with open(log_filename, "a") as outp:
    print >> outp, "Patient=%s\tTotal=%d\tOmittedIntervals=%d\tAmpMin=%d\tAmpMax=%d\tIntMin=%d\tIntMax=%d" % (patient, 
                                                                                                              total,
                                                                                                              omitted_intervals, 
                                                                                                              amplitude_min, 
                                                                                                              amplitudes_max, 
                                                                                                              interval_min, 
                                                                                                              interval_max)
  print "Omitted=%d\tTotalCorrected=%d\tTotalInput=%d" % (omitted_intervals, total, len(q_samples))
  

log_filename=sys.argv[1]
patient = sys.argv[2]
local_path = "ptbdb/"

hd_path = local_path
if len(sys.argv) > 3:
  hd_path = sys.argv[3] + "/"

signal = "0"
if len(sys.argv) > 4:
  signal = sys.argv[4]

data_filename = hd_path + patient
qrs_input_filename = hd_path + patient + "_sqrs_output_" + signal

qrs_output_filename = local_path + patient + "_qrs_"
corrected_qrs_output_filename = qrs_output_filename + "corrected_" + signal
qrs_output_filename = qrs_output_filename + signal

lines_count = len([x for x in open(data_filename) if len(split_line(x)) > 1 + int(signal)])

mode = "ONE"
if len(sys.argv) > 5:
  mode = sys.argv[5]

noise_by_window_size_filename = "noise_by_window_size_" + signal
noise_filename = local_path + patient + "_noise_" + signal
"""
process_noise(corrected_qrs_output_filename, data_filename, lines_count, patient, \
              noise_by_window_size_filename, noise_filename, signal = int(signal), mode = mode)
"""
calculate_r_peaks(data_filename, lines_count, qrs_input_filename, \
                  qrs_output_filename, corrected_qrs_output_filename, \
                  patient, log_filename, 
                  signal = int(signal))
