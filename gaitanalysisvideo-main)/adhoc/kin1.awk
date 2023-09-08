# generate shell commands to extract a segment of kinematics data within the midpoint gait cycle
# Optional parameter: 
## oLeft = custom left midpoint gait cycle
## oRight = custom right midpoint gait cycle
# sample input with space delimiter:
## run_30-11-2022_16-07-10.log:Left strides (full gait cycles): 3
## run_30-11-2022_16-07-10.log-[[182, 290], [290, 418], [418, 533]]
# sample output:
## awk 'NR==248+2, NR==374+2 {print FILENAME "," $0}' *02-11-2022_15-44-59*_kine_1.csv
## awk 'NR==190+2, NR==310+2 {print FILENAME "," $0}' *02-11-2022_15-44-59*_kine_2.csv

/Left/ {
  side=1
  if (oLeft > 0) midpoint = oLeft; else midpoint = int(($6+1)/2)
}
/Right/ {
  side=2
  if (oRight > 0) midpoint = oRight; else midpoint = int(($6+1)/2)
}
/full gait/ {
  patsplit($1, filename, /[0-9][0-9\-_]+/)
  next}
{
 patsplit($0, pair, /\[[0-9, ]+\]/)
 #print filename[1], side, pair[midpoint]
 patsplit(pair[midpoint], range, /[0-9]+/)
 printf("awk 'NR==%d+2, NR==%d+2 {print FILENAME \",\" $0}' *%s*_kine_%d.csv\n", range[1], range[2], filename[1], side)
}
