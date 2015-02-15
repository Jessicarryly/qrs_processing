HOME_PATH="/Users/shushu/WFDB/"
PY_PATH="/Users/shushu/Documents/WFDB/qrs_processing/py"

cd $HOME_PATH


DIAGNOSES_FILE="diagnoses_list.txt"
OUT_FILE="patients_count.txt"
rm $OUT_FILE

exec 3<"$DIAGNOSES_FILE"
while read -r diagnosis <&3
do
  echo $diagnosis
  FILE="diagnosis/${diagnosis}"
  python ${PY_PATH}/count_patients.py $FILE $diagnosis $OUT_FILE
done
