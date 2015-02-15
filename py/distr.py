from collections import defaultdict
import os
os.chdir("/Users/shushu/Documents/WFDB/")


distrs = defaultdict(int)
for line in open("lengths.txt"):
  distrs[line.split("\t")[1].strip()] += 1

with open("distrs.txt", "w") as distrs_file:
  length_sorted_freq = sorted([(key, freq) for key, freq in distrs.iteritems()], 
                              key = lambda x: x[1])
  for key, freq in length_sorted_freq:
    print >> distrs_file, "%s\t%d" % (key, freq)
  
  
