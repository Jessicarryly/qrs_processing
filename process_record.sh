
record="patient291/s0554_re"
signal="2"
to_inv="NOT"
threshold="260"

python py/invert_signal_by_sign.py /Volumes/WD500GB/WFDB/ptbdb_muv $record $signal $to_inv $threshold
ec=""
ec=$(python py/get_noisy_segments.py ~/Documents/WFDB/log/log $record /Volumes/WD500GB/WFDB/ptbdb_muv $signal)
python py/plot_record.py $record $signal $ec

