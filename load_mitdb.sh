loadFromDatabase() {
  
  if ! [ -a ${path}/$patient ]; then
    mkdir ${path}/$patient
  fi
  
  muv_filename="${patient}/${patient}"
  echo $muv_filename
  #rdsamp -r mitdb/${patient} -P -PS -H > ${path}/${muv_filename}_copy
  #python ${PY_PATH}/replace_spaces_with_tabs.py $muv_filename 1000 mitdb
  
  #rdann -r mitdb/$patient -a atr -v > ${path}/${patient}/${patient}_rpeaks
}

PY_PATH="/Users/shushu/Documents/WFDB/qrs_processing/py"
patients_list="/Volumes/WD500GB/WFDB/mitdb/records"
path="/Volumes/WD500GB/WFDB/mitdb/mitdb"

exec 3<"$patients_list"

while read -r patient <&3
do
  echo $patient
  loadFromDatabase
done