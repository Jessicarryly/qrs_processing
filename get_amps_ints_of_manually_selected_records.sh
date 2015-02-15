HOME_PATH="/Users/shushu/WFDB/"
PY_PATH="/Users/shushu/Documents/WFDB/qrs_processing/py"

cd $HOME_PATH


getQRS() { 
  info=$(python ${PY_PATH}/get_noisy_segments.py log/${log}_${signal_index} $patient_record $path $signal_index ONE)
  echo $info
}
  

timestamp() {
  date +"%Y%m%d_%H%M"
}

 
path="/Volumes/WD500GB/WFDB/ptbdb"

<<comment1
TO_INVERT="/Volumes/WD500GB/WFDB/to_invert"
exec 3<"$TO_INVERT"
while read -r patient_record_signal <&3
do
  echo $patient_record_signal

  OIFS=$IFS
  IFS=' '
  read -ra patient_record_signal <<< "${patient_record_signal}"
  IFS=$OIFS

  patient_record="${patient_record_signal[0]}/${patient_record_signal[1]}"
  signal_index="${patient_record_signal[2]}"
  mv ${path}/${patient_record}_sqrs_output_${signal_index} ${path}/${patient_record}_sqrs_output_${signal_index}_before_inverting 
  mv ptbdb/${patient_record}_qrs_corrected_${signal_index} ptbdb/${patient_record}_qrs_corrected_${signal_index}_before_inverting 
  python ${PY_PATH}/invert_signal_by_sign.py $path $patient_record $signal_index
done  

python ${PY_PATH}/prepare_for_editing_sqrs_corrected.py

log="log_$(timestamp)"
exec 3<"$TO_INVERT"
while read -r patient_record_signal <&3
do
  echo $patient_record_signal

  OIFS=$IFS
  IFS=' '
  read -ra patient_record_signal <<< "${patient_record_signal}"
  IFS=$OIFS

  patient_record="${patient_record_signal[0]}/${patient_record_signal[1]}"
  signal_index="${patient_record_signal[2]}"
  getQRS
done  
#python plot_ecg.py log/log_20150205_0307 invert
comment1


<<comment1
processDiagnosis() { 
  echo $patients_list
  exec 0<"$patients_list"
  hd_path="/Volumes/WD500GB/WFDB/ptbdb/"
  while read -r line
  do
    echo $line
    #python ${PY_PATH}/replace_spaces_with_tabs.py $line

    for signal_index in {0..2}
    do
      #echo $signal_index
      #if [ -a ${hd_path}/${line}_sqrs_output_${signal_index}_before_inverting ]; then
      #  mv ${hd_path}/${line}_sqrs_output_${signal_index}_before_inverting ${hd_path}/${line}_sqrs_output_${signal_index}
      #fi
      #if [ -a ptbdb/${line}_qrs_corrected_${signal_index}_before_inverting ]; then
      #  mv ptbdb/${line}_qrs_corrected_${signal_index}_before_inverting ptbdb/${line}_qrs_corrected_${signal_index}
      #fi
      
    #done
    
  done
} 

DIAGNOSES_FILE="diagnoses_list.txt"
exec 3<"$DIAGNOSES_FILE"
while read -r diagnosis <&3
do
  echo $diagnosis
  patients_list="diagnosis/${diagnosis}"
  processDiagnosis  
  #getDiffs
done
comment1


prepareFolders() {
  if [ $MODE == "SELECTED" ]; then 
    if [ $diagnosis == "healthy_control" ]; then
      diff_output_path="diagnosis/healthy_selected"
    else
      diff_output_path="diagnosis/sick_selected"
    fi
  else 
    if [ $diagnosis == "healthy_control" ]; then
      diff_output_path="diagnosis/healthy"
    else
      diff_output_path="diagnosis/sick"
    fi
  fi

  if ! [ -d $diff_output_path ]; then
    mkdir $diff_output_path
  fi
}



printReady() {
  READY="$1"
  exec 3<"$READY"
  while read -r patient_record_signal <&3
  do
    OIFS=$IFS
    IFS=' '
    read -ra patient_record_signal <<< "${patient_record_signal}"
    IFS=$OIFS
    patient_record="${patient_record_signal[0]}/${patient_record_signal[1]}"
    signal_index="${patient_record_signal[2]}"
    diagnosis="${patient_record_signal[3]}"
    prepareFolders
    python ${PY_PATH}/print_amps_intervals_by_diagnosis.py $patient_record $diff_output_path $signal_index
  done
}


#python ${PY_PATH}/append_diagnosis.py
READY_ALL="/Volumes/WD500GB/WFDB/ready"
MODE="ALL"


python ${PY_PATH}/select_signal_ready.py
READY_SELECTED="/Volumes/WD500GB/WFDB/ready_selected"
MODE="SELECTED"

if [ $MODE == "SELECTED" ]; then 
  printReady $READY_SELECTED
else
  printReady $READY_ALL
fi

