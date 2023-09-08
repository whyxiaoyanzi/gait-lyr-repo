# create shell commands to copy files to web folder

# sample command to generate webfilelist:
## ls -1 *_gait.csv *_kine_?.csv *_cbta_?.csv > webfilelist
# sample webfilelist:
## AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR_cbta_1.csv
## AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR_gait.csv
## AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR_kine_1.csv

# sample commnd to generate webdirlist:
## find /data/pdlogger/storage/app/public -type d > webdirlist
# sample webdirlist:
## /data/pdlogger/storage/app/public/AR Gait_P_HS2809202201_Free walk_28-09-2022_11-11-35
## /data/pdlogger/storage/app/public/AR Gait_P_HS0909202202_Free walk_09-09-2022_17-10-48
## /data/pdlogger/storage/app/public/AR Gait_P_P00002S_Free walk_15-05-2023_16-15-43

while read filename
do
 fname=`basename "$filename"`
#echo fname = $fname
 vname=${fname%_noAR*}
#echo vname = $vname
 dname=`grep "${vname}" webdirlist`
#echo dname = $dname
 echo cp \"$filename\" \"$dname\"
done < webfilelist
