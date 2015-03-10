import sys
import numpy as np
#!!!!!
#!!!!!TODO change functions names according to camelCase!!!!!
#!!!!!
from base_functions import getSignalValues, getSplittedLine, strOfList, changeSFreq, getRobustMedianAndStdev 
import os
from matplotlib import pyplot as plt
import tempfile
from shutil import copy, copyfile
from get_qrs_peaks_comparison import calculateRPeaks

def constraintValueBySegment(value, min_value, max_value):
  value = max(value, min_value)
  value = min(value, max_value)
  return value

def improveSlopecrit(nslope, slopecrit, scmin, scmax):
  if nslope == 0:
    slopecrit -= slopecrit / 16
  elif nslope >= 5:
    slopecrit += slopecrit / 16
  return constraintValueBySegment(slopecrit, scmin, scmax)

def updateWindow(window_values, signal_value, window_size = 10):
  window_values.append(signal_value)
  if len(window_values) == 1:
    window_values = [signal_value, ] * 10
  window_values = window_values[-10:]
  return window_values

def calculateSlope(window_values):
  return window_values[0] - window_values[-1] + \
         4 * (window_values[1] - window_values [-2]) + \
         6 * (window_values[2] - window_values[-3]) + \
         4 * (window_values[3] - window_values[-4]) + \
         window_values[4] - window_values[-5]


def checkIfNewSlope(sclope, slope_sign, nslope, maxtime, ms200, ms160):
  slope_sign = -slope_sign
  nslope += 1
  if nslope > 4:
    maxtime = ms200
  else:
    maxtime = ms160
  return slope_sign, nslope, maxtime

def correctQSample(qsample, corrected_qsamples_set):
  corrected_qsamples_set = np.array(corrected_qsamples_set)
  qsample_surround = \
    corrected_qsamples_set[corrected_qsamples_set[:, 0] >= qsample - 40, :]
  #print qsample, min(qsample_surround[:, 0]), max(qsample_surround[:, 0])

  qsample_surround = qsample_surround[qsample_surround[:, 0] <= qsample - 5, :]
  if qsample > 5:
    corrected_qsample = qsample_surround[np.argmin(qsample_surround[:, 1]), 0]
  else:
    corrected_qsample = qsample

  qsample_surround = \
    corrected_qsamples_set[abs(corrected_qsamples_set[:, 0] - qsample) <= 40, :]

  qsample_left_surround = qsample_surround[qsample_surround[:, 0] <= qsample - 4, :]
  qsample_right_surround = qsample_surround[qsample_surround[:, 0] >= qsample - 4, :]

  right_qsample = qsample_right_surround[0, 0]  
  for i in xrange(1, len(qsample_right_surround)):
    if qsample_right_surround[i, 1] <= qsample_right_surround[i - 1, 1]:
      right_qsample = qsample_right_surround[i, 0]
    else:
      break
  
  if right_qsample > qsample_right_surround[0, 0]:
    corrected_qsample = right_qsample
  else:
    left_qsample = qsample_left_surround[-1, 0]
    for i in xrange(len(qsample_left_surround) - 2, -1, -1):
      if qsample_left_surround[i, 1] <= qsample_left_surround[i + 1, 1]:
        left_qsample = qsample_left_surround[i, 0]
      else:
        break
    corrected_qsample = left_qsample
  return corrected_qsample

def printQSample(slopecrit, maxslope, scmin, scmax, samples_to_last_qpeak, 
                  qsample, sample, samples_multiplier, nslope, output_file):
  if 2 <= nslope:
    if nslope <= 4:
      qtype = "N"
      slopecrit += (maxslope / 4 - slopecrit) / 8
      slopecrit = constraintValueBySegment(slopecrit, scmin, scmax)

      samples_to_last_qpeak = 0
    else:  
      qtype = "A"  
    qsample = max((qsample) * samples_multiplier, 0)
    sample = sample * samples_multiplier
    print >> output_file, "%d\t%s" % (int(qsample), qtype) 
  return slopecrit, samples_to_last_qpeak

def processSample(line, signal_index, sfreq, inp_sfreq, outp,
                   window_values, slope_params, qsample_params):
  slopecrit, scmin, scmax, nslope, slope_sign, maxslope= slope_params
  maxtime, samples_to_last_qpeak, qsample, corrected_qsamples_set = qsample_params

  ms200, ms160, s2 = 0.2 * sfreq, 0.16 * sfreq, 2 * sfreq

  sample, signal_value = int(line[0]), float(line[signal_index + 1])
  window_values = updateWindow(window_values, signal_value)

  corrected_qsamples_set.append((sample, signal_value))
  slope = calculateSlope(window_values)
 
  samples_to_last_qpeak += 1
  if samples_to_last_qpeak % s2 == 0:
    slopecrit = improveSlopecrit(nslope, slopecrit, scmin, scmax)

  if nslope == 0 and abs(slope) > slopecrit:
    nslope = 1
    slope_sign = np.sign(slope)
    qsample = sample 
    maxtime = ms160

  if nslope != 0:
    if slope * slope_sign < -slopecrit:
      slope_sign, nslope, maxtime = checkIfNewSlope(slope, slope_sign, nslope,  
                                                       maxtime, ms200, ms160)
    elif slope * slope_sign > slopecrit and abs(slope) > maxslope: 
      maxslope = abs(slope)

    if maxtime < 0: 
      qsample = correctQSample(qsample, corrected_qsamples_set)
      slopecrit, samples_to_last_qpeak = \
            printQSample(slopecrit, maxslope, scmin, scmax, samples_to_last_qpeak, 
                         qsample, sample, inp_sfreq * 1.0 / sfreq, nslope, outp)
      corrected_qsamples_set = [(x, y) for (x, y) in corrected_qsamples_set if x >= qsample]
      nslope = 0
    maxtime -= 1
  slope_params = [slopecrit, scmin, scmax, nslope, slope_sign, maxslope]
  qsample_params = [maxtime, samples_to_last_qpeak, qsample, corrected_qsamples_set]
  return window_values, slope_params, qsample_params


def getQPeaks(datadir, record, resampledir, outdir,
              q_suffix, inp_sfreq, 
              muv_scmin, signal_index= 0, ADC= False):
  scmin = muv_scmin
  if ADC == True:
    scmin *= 2

  sfreq = 250
  """
  if sfreq != inp_sfreq:
    changeSFreq(datadir + record, resampledir + record, inp_sfreq, out_sfreq= sfreq, first_signal_index= 1)
  else:
    copy(datadir + record, resampledir)
  """
  with open(resampledir + record) as inp, \
       open(outdir + record + q_suffix + str(signal_index), "w") as outp:
    signal_index = int(signal_index)
    
    slopecrit = scmax = 10 * scmin
    nslope = maxslope = maxtime = samples_to_last_qpeak = qsample = 0
    slope_sign = 1
    window_values = []
    corrected_qsamples_set = []

    slope_params = [slopecrit, scmin, scmax, nslope, slope_sign, maxslope]
    qsample_params = [maxtime, samples_to_last_qpeak, qsample, corrected_qsamples_set]

    for line in inp:
      line = getSplittedLine(line)
      window_values, slope_params, qsample_params = \
            processSample(line, signal_index, sfreq, inp_sfreq, outp,
                          window_values, slope_params, qsample_params)
      
def plotQPeaksInWindow(dataset, peaks_list, left, window_size, ax):
  right = left + window_size
  ax.plot(dataset[left : right, 0], dataset[left : right, 1])
  for peaks in peaks_list:
    ax.plot(peaks[:, 0], peaks[:, 1], 'o')
  ax.set_xlim((left, right))

def plotQPeaks(datadir, qrsdir, plotsdir, record, qrs_suffix, inp_sfreq, peaks_count= 1, signal_index= 0):
  signal_index = str(signal_index)
  dataset = np.loadtxt(datadir + record)

  peaks_list = []
  for i in xrange(peaks_count):
    peaks = []
    with open(qrsdir + record + qrs_suffix + signal_index) as inp:
      for line in inp:
        line = getSplittedLine(line)
        peak = int(line[i])
        peaks.append((peak, dataset[peak, 1 + int(signal_index)]))
    peaks_list.append(np.array(peaks))

  diff = 5 * inp_sfreq 
  plt.clf()
  subplots_count = 4
  f, axes = plt.subplots(subplots_count, sharey= False, figsize= (20, 10))
  left = 2800
  for ax in axes:
    plotQPeaksInWindow(dataset, peaks_list, left, diff, ax)
    left += diff
  f.savefig(plotsdir + record + qrs_suffix + signal_index + ".png")

def getCoeffs(index):
  """
  if index == 0:
    return 1
  else:
    return 0
  """
  return 1.0 / (index + 1)

def calculateNextSlope(window_values):
  window_size = len(window_values)
  slope = 0
  for index in xrange(window_size / 2):
    slope += (window_values[-index - 1] - window_values[index]) * getCoeffs(index)
  return slope

def findNextPeak(previous_peak, slope_sign, dataset, sfreq, signal_index= 0):
  if sfreq < 100:
    window_size = 2
    first_sample = 1
  elif sfreq < 1000:
    window_size = 30
    first_sample = 25
  if previous_peak + 1 < window_size:
    sample = min(previous_peak + window_size, len(dataset))
  else:
    sample = min(previous_peak + first_sample, len(dataset))
  slope = calculateNextSlope(dataset[sample + 1 - window_size : sample + 1, signal_index + 1])
  while np.sign(slope) == slope_sign and sample <= len(dataset):
    sample += 1
    window_values = dataset[sample + 1 - window_size : sample + 1, signal_index + 1]
    slope = calculateNextSlope(window_values)
  while sample >= len(dataset) or \
        ((dataset[sample - 1, signal_index + 1] - dataset[sample, signal_index + 1]) * slope_sign) > 0:
    sample -= 1
  while sample < len(dataset) - 1 and \
        ((dataset[sample + 1, signal_index + 1] - dataset[sample, signal_index + 1]) * slope_sign) > 0:
    sample += 1
  return sample

def getQRSPeaks(datadir, qrsdir, record, q_suffix, qrs_suffix, inp_sfreq, signal_index= 0):
  dataset = np.loadtxt(datadir + record)
  qpeaks = []
  with open(qrsdir + record + q_suffix + str(signal_index)) as qpeaks_input:
    for line in qpeaks_input:
      qpeak = int(getSplittedLine(line)[0])
      qpeaks.append(qpeak)
  with open(qrsdir + record + qrs_suffix + str(signal_index), "w") as qrs_output:
    qrs_window = []
    for qpeak in qpeaks:
      rpeak = findNextPeak(qpeak, 1, dataset, inp_sfreq, signal_index)
      speak = findNextPeak(rpeak, -1, dataset, inp_sfreq, signal_index)
      print >> qrs_output, "%d\t%d\t%d" % (qpeak, rpeak, speak)

def getNamesToSfreqsMap(data_path):
  names_to_sfreqs_map = {}
  with open(data_path + "name_to_sfreq_map.txt") as f:
    for line in f:
      line = getSplittedLine(line)
      print line
      names_to_sfreqs_map[line[0]] = int(line[1])
  return names_to_sfreqs_map

def getDeviceRoot(device_type):
  #root = "/Volumes/ShuSilicon/WFDB/comparison/"
  root = "/Users/shushu/Documents/WFDB/comparison/"
  if device_type == "TEST":
    root += "test_device/"
  else:
    root += "skrinfax/"
  return root

def addSample(datadir, record):
  copyfile(datadir + record, datadir + record + "_before_inv")
  with open(datadir + record, "w") as outp, open(datadir + record + "_before_inv") as inp:
    for line in inp:
      line = getSplittedLine(line)
      line[1] = str(-1 * float(line[1]))
      print >> outp, "\t".join(line)
      
good = {
  "TEST" : ["4_2.txt", "5_1.txt", "6_3.txt"], 
  "SFAX" : ["1_1.txt", "1_2.txt", "2_1.txt", "3_2.txt"]
}

def getCorrectedRAmpsInts(ramps_ints):
  ramps_ints = ramps_ints[ramps_ints[:, 1] > 0, :]

  b = len(ramps_ints)
  ramps_med, ramps_std = getRobustMedianAndStdev(ramps_ints[:, 0])
  print ramps_med, ramps_std, ramps_med - 3 * ramps_std
  ramps_ints = ramps_ints[ramps_med - ramps_ints[:, 0] < 3 * (ramps_std + 10), :]
  print len(ramps_ints), b

  ramps_ints[:, 1] = np.r_[np.diff(ramps_ints[:, -2]), np.median(ramps_ints[:, 1])]
  rints_med, rints_std = np.median(ramps_ints[:, 1]), np.std(ramps_ints[:, 1])

  print rints_med, rints_std, rints_med - 3 * rints_std

  ramps_ints = ramps_ints[abs(ramps_ints[:, 1] - rints_med) < 3 * (rints_std + 2), :]
  print len(ramps_ints), b

  return ramps_ints

def plotRPeaksAndMinOfQS(dataset, rpeaks, mins, figname, signal_index = 0):
  plt.clf()
  plt.plot(dataset[:, 0], dataset[:, 1 + signal_index])
  plt.plot(rpeaks, dataset[rpeaks, 1 + signal_index], 'o')
  plt.plot(mins, dataset[mins, 1 + signal_index], 'o')
  plt.xlim((0, 1000))
  plt.savefig(figname)

def getRAmpsInts(datadir, qrsdir, record, qrs_suffix, signal_index = 0):
  dataset = np.loadtxt(datadir + record)
  peaks = np.loadtxt(qrsdir + record + qrs_suffix + str(signal_index), dtype = (int))
  qsintervals = peaks[:, 2] - peaks[:, 0]
  qramps = dataset[peaks[:, 1], 1 + signal_index] - dataset[peaks[:, 0], 1 + signal_index]
  qramps_med, qramps_std = getRobustMedianAndStdev(qramps)
  qsints_med, qsints_std = getRobustMedianAndStdev(qsintervals)
  qsints_std = int(qsints_std) + 1

  ramps_ints = []
  
  for qrs_index in xrange(peaks.shape[0]):
    q, r, s = peaks[qrs_index]
    qrs_values = dataset[q : s + 1, 1 + signal_index]
    while max(qrs_values) - min(qrs_values) < qramps_med - 2 * qramps_std:
      s += qsints_std + 1
      qrs_values = dataset[q : s + 1, 1 + signal_index]

    rpeak = np.argmax(qrs_values) + q
    ramp = dataset[rpeak, 1 + signal_index] - min(qrs_values)
    min_of_qs = np.argmin(qrs_values) + q
    if qrs_index > 0: 
      ramps_ints.append((previous_ramp, rpeak - previous_rpeak, previous_rpeak, previous_min_of_qs))
    previous_rpeak, previous_ramp, previous_min_of_qs = rpeak, ramp, min_of_qs

  ramps_ints = getCorrectedRAmpsInts(np.array(ramps_ints))
  with open(datadir + record.split(".")[0] + "_ramps_" + str(signal_index) + ".txt", "w") as out:
    for index in xrange(ramps_ints.shape[0]):
      print >> out, "%.3f\t%d" % (ramps_ints[index, 0], ramps_ints[index, 1])

  rpeaks = np.array(ramps_ints[:, -2], dtype = int)
  mins = np.array(ramps_ints[:, -1], dtype = int)
  plotRPeaksAndMinOfQS(dataset, rpeaks, mins, qrsdir + record.split(".")[0] + ".jpg", signal_index)

def getRAmpsIntsCorr(datadir, qrsdir, record, qrs_suffix, peaks_count = 3, signal_index = 0, skip_nrows = 0):
  with open(datadir + record.split(".")[0] + "_ramps_" + str(signal_index) + ".txt", "w") as outp, \
       open(qrsdir + record + qrs_suffix + str(signal_index)) as qrs_peaks_corrected:
    for i in xrange(skip_nrows):
      qrs_peaks_corrected.readline()
    for line in qrs_peaks_corrected:
      ramp, rint = getSplittedLine(line)[peaks_count:peaks_count + 2]
      print >> outp, "%s\t%s" % (ramp, rint)

def processDevice(muv_scmin, DEV_TYPE= "TEST"):
  root = getDeviceRoot(DEV_TYPE)
  os.chdir(root)
  datadir = "data/"
  names_to_sfreqs_map = getNamesToSfreqsMap(datadir)

  resampledir = datadir + "resample/"
  #print resampledir

  signal_index = 0
  qrsdir = "qrs_peaks/"
  plotsdir = "plots/"

  q_suffix = "_q_peaks_"
  qrs_suffix = "_qrs_peaks_"
  """
  qrs_output_filename = record + qrs_suffix + "_with_errors_"
  corrected_qrs_output_filename = record + qrs_suffix + str(signal_index)
  qrs_output_filename = qrs_output_filename + str(signal_index)
  """

  for record, inp_sfreq in names_to_sfreqs_map.iteritems():
    if DEV_TYPE == "TEST" and record not in good[DEV_TYPE]: 
      print DEV_TYPE, record
      """
      getQPeaks(datadir, record, resampledir, qrsdir, q_suffix,
                inp_sfreq, muv_scmin, signal_index, ADC= False)
      
      getQRSPeaks(datadir, qrsdir, record, q_suffix, qrs_suffix, inp_sfreq, signal_index)
      
      plotQPeaks(datadir, qrsdir, plotsdir, record, qrs_suffix, inp_sfreq, peaks_count= 3, signal_index= signal_index) 
      """
      getRAmpsInts(datadir, qrsdir, record, qrs_suffix)



  #os.removedirs(resampledir)
def processDeviceCorr(DEV_TYPE):    
  root = getDeviceRoot(DEV_TYPE)
  os.chdir(root)
  
  log_filename="log"
  
  datadir = "data/"
  names_to_sfreqs_map = getNamesToSfreqsMap(datadir)

  signal_index = 0
  qrsdir = "qrs_corr/"
  plotsdir = "plots_corr/"

  qrs_suffix = "_qrs_peaks_"
  for record, inp_sfreq in names_to_sfreqs_map.iteritems():
    if record in good[DEV_TYPE] or DEV_TYPE == "SFAX":
      print DEV_TYPE, record
      data_filename = datadir + record
      qrs_input_filename = qrsdir + record + "_q_peaks_" + str(signal_index)

      qrs_output_filename = qrsdir + record + qrs_suffix + "with_errors_"
      corrected_qrs_output_filename = qrsdir + record + qrs_suffix + str(signal_index)
      qrs_output_filename = qrs_output_filename + str(signal_index)

      lines_count = len([x for x in open(data_filename) if len(getSplittedLine(x)) > 1 + int(signal_index)])
      
      calculateRPeaks(data_filename, lines_count, qrs_input_filename, \
                      qrs_output_filename, corrected_qrs_output_filename, \
                      record, log_filename, 
                      signal = int(signal_index))
      
      plotQPeaks(datadir, qrsdir, plotsdir, record, qrs_suffix, inp_sfreq, peaks_count = 3)
      getRAmpsIntsCorr(datadir, qrsdir, record, qrs_suffix, peaks_count = 3, signal_index = 0)

def filterRAmpsInts(ramps_ints):
  b = len(ramps_ints)
  ramps_med, ramps_std = getRobustMedianAndStdev(ramps_ints[:, 0])
  ramps_ints = ramps_ints[abs(ramps_med - ramps_ints[:, 0]) < 3 * (ramps_std + 10), :]
  print len(ramps_ints), b

  rints_med, rints_std = np.median(ramps_ints[:, 1]), np.std(ramps_ints[:, 1])
  print rints_med, rints_std, rints_med - 3 * rints_std
  ramps_ints = ramps_ints[abs(ramps_ints[:, 1] - rints_med) < 3 * (rints_std + 2), :]
  print len(ramps_ints), b
  return ramps_ints

#def getCode(amplitude_diff, interval_diff, alpha_diff):

def getCodegrams():
  datadir = "data/"
  signal_index = 0
  for dev_type in ["SFAX", "TEST"]:
    root = getDeviceRoot(dev_type)
    os.chdir(root)
    names_to_sfreqs_map = getNamesToSfreqsMap(datadir)
    for record in names_to_sfreqs_map.iterkeys():
      record = record.split(".")[0]
      print record
      ramps_ints = np.loadtxt(datadir + record + "_ramps_" + str(signal_index) + ".txt") 
      
      #ramps_ints = filterRAmpsInts(ramps_ints)
      """
      with open(datadir + record + "_codegrams_" + str(signal_index) + ".txt") as outp:
        for index in xrange(1, len(ramps_ints)):
          pass
      """
if __name__ == "__main__":   
  muv_scmin = 320     
  """
  ptbdb_path = "/Volumes/WD500GB/WFDB/ptbdb/"
  record = "patient006/s0064lre"
  suffix = "_resampled"
  inp_sfreq = 1000
  signal_index = 2
  get_q_peaks(ptbdb_path, record, suffix, inp_sfreq, muv_scmin, signal_index, ADC= True)
  record = record + "_muv"
  get_q_peaks(ptbdb_path, record, suffix, inp_sfreq, muv_scmin, signal_index, ADC= False)
  """
  #DEV_TYPE = "SFAX"
  DEV_TYPE = "TEST"
  processDevice(muv_scmin, DEV_TYPE)
  #processDeviceCorr(DEV_TYPE)
  #processDevice(muv_scmin)
  getCodegrams()

  
