<<comment1
diagnosis="$1"
noise_type=("amp" "int")
for i in "${noise_type[@]}"
do 
  path="../diagnosis/${diagnosis}_stats/"
  file="${path}noisy_by_$i"
  exec 0<"$file"
  while read -r line
  do
    rdsamp -r ptbdb/${line} -H -s 0 > ${path}data
    #python py/plot_ecg.py $diagnosis $i $line
    echo $line
  done
done
comment1

FILE="../diagnosis/healthy_control_copy"
exec 0<"$FILE"
while read -r line
do
  rdsamp -r ptbdb/${line} -H -s 0 > ../ptbdb/${line}
  echo ${line}
done
