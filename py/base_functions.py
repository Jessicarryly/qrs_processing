import re
import numpy as np
from fractions import gcd


def getSignalValues(line, first_signal_index = 1, multiplier = 1):
  return np.array(map(float, line[first_signal_index:])) * multiplier

def getSplittedLine(line):
  return re.split("\s+", line.strip())

def strOfList(values, delim = "\t"):
  return delim.join(map(str, values))

def interpolateValuesByLeftAndRight(left_in_sample_index, right_in_sample_index, 
                                    left_in_sample_values, right_in_sample_values,
                                    out_ticks,
                                    out_sample_index, out_sample_seq_index, out_file):
  while left_in_sample_index <= out_sample_index < right_in_sample_index:

    out_sample_values = (right_in_sample_index - out_sample_index) * left_in_sample_values + \
                        (out_sample_index - left_in_sample_index) * right_in_sample_values
    out_sample_values /= right_in_sample_index - left_in_sample_index
    
    print >> out_file, "%d\t%s" % (out_sample_seq_index, strOfList(out_sample_values))

    out_sample_index += out_ticks
    out_sample_seq_index += 1
  return out_sample_index, out_sample_seq_index


def changeSFreq(in_filename, out_filename, in_sfreq, 
                out_sfreq = 250, first_signal_index = 1):
  with open(in_filename) as inp, \
       open(out_filename, "w") as outp:

    sfreqs_gcd = gcd(in_sfreq, out_sfreq)
    in_ticks = out_sfreq / sfreqs_gcd
    out_ticks = in_sfreq / sfreqs_gcd
    #print in_ticks, out_ticks

    left_in_sample_index = 0

    line = getSplittedLine(inp.readline())
    left_in_sample_values = getSignalValues(line, first_signal_index)
    print >> outp, "%d\t%s" % (0, strOfList(left_in_sample_values))

    out_sample_index = out_ticks
    out_sample_seq_index = 1
    for line in inp:
      line = getSplittedLine(line)
      
      right_in_sample_index = left_in_sample_index + in_ticks
      right_in_sample_values = getSignalValues(line, first_signal_index)

      out_sample_index, out_sample_seq_index = interpolateValuesByLeftAndRight(left_in_sample_index, 
                                                                               right_in_sample_index,
                                                                               left_in_sample_values,
                                                                               right_in_sample_values,
                                                                               out_ticks,
                                                                               out_sample_index, 
                                                                               out_sample_seq_index, 
                                                                               outp)
      left_in_sample_index, left_in_sample_values = right_in_sample_index, right_in_sample_values
 
def getRobustMedianAndStdev(array_values, std_is_modified = True):
  median = np.median(array_values)
  if std_is_modified:
    stdev = np.sqrt(np.median(abs(array_values - median) ** 2)) 
  else:
    stdev = np.sqrt(np.mean(abs(array_values - median) ** 2))
  return median, stdev 

