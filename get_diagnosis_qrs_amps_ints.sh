#!/bin/bash

HOME_PATH="/Users/shushu/WFDB/"
PY_PATH="/Users/shushu/Documents/WFDB/qrs_processing/py"

<<comment1
FILE="all_records.txt"
#FILE="diagnosis/healthy_control"
exec 0<"$FILE"

if ! [-d diagnosis]; then
  mkdir diagnosis
fi 

cd $HOME_PATH
rm header.txt
rm healthy_records.txt
while read -r line
do
  #rm record.txt
  curl -o header.txt http://physionet.org/physiobank/database/ptbdb/${line}.hea  
  python ${PY_PATH}/filter_one_diagnosis.py $line 
  echo $line
done
comment1


getQRS() {
  signal_log="$1"  

  OIFS=$IFS
  IFS='/'
  read -ra patient <<< "${line}"
  IFS=$OIFS
  
  patient="${patient[0]}"
  path="/Volumes/WD500GB/WFDB/ptbdb"

  if ! [ -a ${path}/$patient ]; then
    mkdir ${path}/$patient
  fi

  cd $HOME_PATH
  #rdsamp -r ptbdb/${line} -H -s 0 1 2 > ${path}/${line}_copy
  #python ${PY_PATH}/replace_spaces_with_tabs.py $line
  #echo $signal_index
  #sqrs -r ptbdb/${line} -H -m 200 -s $signal_index
  #rdann -r ptbdb/${line} -a qrs > ${path}/${line}_sqrs_output_${signal_index}
  #info=$(python ${PY_PATH}/get_amplitudes.py log/$signal_log $line $path)

  #info=$(python ${PY_PATH}/get_amplitudes_new.py log/$signal_log $line $path MULT $noise_by_window_size)
  info=$(python ${PY_PATH}/get_noisy_segments.py log/${signal_log} $line $path $signal_index ONE $noise_by_window_size)
  echo $info
  findErrorsInQRSRecognition $line $info 
}
  

findErrorsInQRSRecognition() {
  patient="$1"
  omitted="$2"
  total_corrected="$3"
  total_input="$4"
  
  OIFS=$IFS
  IFS='='
  read -ra omitted  <<< "$omitted"
  read -ra total_corrected <<< "$total_corrected"
  read -ra total_input <<< "$total_input"
  IFS=$OIFS
  omitted=${omitted[1]}
  total_corrected=${total_corrected[1]}
  total_input=${total_input[1]}
  if ! [ -a log/errors_${signal_log} ]; then
    echo -e "Patient\tOmitted\tTotalCorrected\tTotalInput" >> log/errors_${signal_log}
  fi

  if ((${omitted} > 0)) || ((${total_corrected} < 4)) || ((${total_input} > ${total_corrected} + 1)); then
    output="${patient}\t${omitted}\t${total_corrected}\t${total_input}"
    echo -e "${output}" >> log/errors_${signal_log} 
    if ((${omitted} > 0)); then  
      echo -e "${output}" >> log/omitted_${signal_log}
    elif ((${total_corrected} < 4)); then
      echo -e "${output}" >> log/none_corrected_${signal_log}
    elif ((${total_input} > ${total_corrected} + 1)); then
      echo -e "${output}" >> log/less_corrected_${signal_log}
    fi
  fi
}


compareFileLengths() {
  line="$1"
  QRS_COUNT=$(wc -l < ptbdb/${line}_qrs_corrected)
  Q_COUNT=$(wc -l < ptbdb/${line}_qrs)
  if ((${QRS_COUNT} < ${Q_COUNT})); then
    echo "Not equal!!! ${line}"
  fi
}

getSubsetProportion() {
  diagnosis="$1"
  if [ "$diagnosis" == "healthy_control" -o "$diagnosis" == "myocardial_infarction" ]; then
    part=0.1
  else
    part=0.3
  fi
}

prepareFolders() {
  if [ $diagnosis == "healthy_control" ]; then
    diff_output_path="diagnosis/healthy"
  else
    diff_output_path="diagnosis/sick"
  fi
  if ! [ -d $diff_output_path ]; then
    mkdir $diff_output_path
  fi
}

getDiffs() {
  prefix="log/selected_signals"
  selected_signal=$(python ${PY_PATH}/select_signal.py $line)
  echo $selected_signal

  if ! [ -a ${stats} ]; then
    echo -e "Record\tNoiseToAmpStd\tRPeaksCnt\tAmpStd\tNoiseStd\tIntStd" >> ${stats}
  fi
  python ${PY_PATH}/get_differencies.py $line ${stats} $diff_output_path $selected_signal
}


processDiagnosis() {
  diff_output_path="."
  prepareFolders 

  echo $patients_list
  exec 0<"$patients_list"
  while read -r line
  do
    echo $line
    #for signal_index in {0..2}
    #do
      #echo $signal_index
      #getQRS ${log}_${signal_index}
    #done
    #getDiffs
  done
}  


timestamp() {
  date +"%Y%m%d_%H%M"
}


cd $HOME_PATH

if ! [ -d log ]; then 
  mkdir log
fi


log="log_$(timestamp)"
stats="diagnosis/stats"
#if [ -e ${stats} ]; then 
#  rm $stats
#fi

noise_by_window_size="noise_by_window_size"
#rm $noise_by_window_size

line_seed=3
part=0.1

DIAGNOSES_FILE="diagnoses_list.txt"
exec 3<"$DIAGNOSES_FILE"
while read -r diagnosis <&3
do
  echo $diagnosis
  line_seed=$((line_seed + 1))
  #getSubsetProportion $diagnosis  
  #python ${PY_PATH}/get_subset_of_file.py diagnosis/${diagnosis} diagnosis/subset/${diagnosis} $part $line_seed
  patients_list="diagnosis/${diagnosis}"
  processDiagnosis  
  #getDiffs
done
python ${PY_PATH}/plot_ecg.py log/log_20150117_0449 plots
  
