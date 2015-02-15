# Set loop separator to end of line
FILE="../all_records.txt"
rm ../lengths.txt
BAKIFS=$IFS
IFS=$(echo -en "\n\b")
exec 3<&0
exec 0<"$FILE"
while read -r line
do
  rm ../record.txt
  wfdbdesc "ptbdb/$line" > ../record.txt
  python py/get_lengths.py
  echo $line
done
exec 0<&3
   
# restore $IFS which was used to determine what the field separators are
IFS=$BAKIFS
exit 0
