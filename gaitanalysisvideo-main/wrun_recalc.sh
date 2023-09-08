#!/usr/bin/bash
# Usage    : Used by wsubmit.sh to process 1 video file
# Parameter: 1) File name (e.g /data/pdlogger/storage/app/public/ARGait_P_H019_Free walk_09-12-2022_15-24-52/ARGait_P_H019_Free walk_09-12-2022_15-24-52_noAR.pkl)
#            2) Left Leg Length in mm
#            3) Right Leg Length in mm
#            4) Params
#            5) override left stride (0 = default midpoint)
#            6) override right stride (0 = default midpoint)
# Output   : _gait.csv, _kine_?.csv, _cbta_?.csv

echo Started at `date`

OPENPOSE_ROOT=/data/openpose
WEB_ROOT=/data/pdlogger
WEB_PUBLIC=$WEB_ROOT/public/PD
pkl=${1}
videoName=${pkl#/*/*/*/*/*/*/}
videoName=${videoName%.pkl}
OLEFT=$5
ORIGHT=$6
if [ ! -f "$pkl" ]; then
 echo ERROR: Input file $pkl not found
 exit 1
fi

# if output directory exist, clear the content else create directory
if [ -d "$OPENPOSE_ROOT/output/$videoName" ]
then
 rm -f "$OPENPOSE_ROOT/output/$videoName/*"
else
 mkdir "$OPENPOSE_ROOT/output/$videoName"
fi

# set env variables for python scripts
export PYTHONPATH=.:$OPENPOSE_ROOT/gait:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/python/openpose
export MODEL_DIR=$OPENPOSE_ROOT/openpose/models
export LD_LIBRARY_PATH=$OPENPOSE_ROOT/openpose/build/python/openpose:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/src/openpose:$LD_LIBRARY_PATH

cd "$OPENPOSE_ROOT/output/$videoName"
python3 $OPENPOSE_ROOT/gait/adhoc/gaitanalysis.py "$pkl" $2 $3 "$4"
# if [ $? -gt 0 ]; then
#  echo ERROR: Gait analysis failed
#  exit 1
# fi

# gather gait analysis data
cbtaName="${videoName}_cbta.csv"
savName="${videoName}_sav.csv"
gaitName="${videoName}_gait.csv"
kineName="${videoName}_kine.csv"
imuName="${videoName}_imu.csv"
frontName="${videoName}_front.csv"
echo cbta > "$gaitName"
if [ -f "$cbtaName" ]; then
 awk -f $OPENPOSE_ROOT/gait/pickcsv.awk oLeft=$OLEFT oRight=$ORIGHT "$cbtaName" >> "$gaitName"
else
 echo No result >> "$gaitName"
fi
echo >> "$gaitName"
echo front >> "$gaitName"
if [ -f "$frontName" ]; then
 awk -f $OPENPOSE_ROOT/gait/pickcsv.awk oLeft=$OLEFT oRight=$ORIGHT "$frontName" >> "$gaitName"
else
 echo No result >> "$gaitName"
fi
echo >> "$gaitName"
echo sav >> "$gaitName"
if [ -f "$savName" ]; then
 awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$savName" >> "$gaitName"
else
 echo No result >> "$gaitName"
fi
echo >> "$gaitName"
echo IMU >> "$gaitName"
if [ -f "$imuName" ]; then
 awk -f $OPENPOSE_ROOT/gait/pickcsv.awk "$imuName" >> "$gaitName"
else
 echo No result >> "$gaitName"
fi
echo >> "$gaitName"
echo kine >> "$gaitName"
if [ -f "$kineName" ]; then
 cat "$kineName" >> "$gaitName"
else
 echo No result >> "$gaitName"
fi
if [ $(grep -c "No result" *gait.csv) -eq 5 ]; then
 rm "$gaitName"
fi

# copy the files to Web public PD folder
cp -f *_gait.csv "$WEB_PUBLIC/${videoName%_noAR}"
cp -f *_kine_?.csv "$WEB_PUBLIC/${videoName%_noAR}"
cp -f *_cbta_?.csv "$WEB_PUBLIC/${videoName%_noAR}"

# update pd_assessment table with gait parameters
if [ -f "$gaitName" ]; then
 fieldSets=`awk -F',' -f $WEB_ROOT/supporting.files/update_pda.awk "$gaitName"`
 php $WEB_ROOT/artisan pda:update "${videoName%_noAR}" "$fieldSets"
 fieldSets=`echo $4 | tr \" \# | awk '{printf("{\"runParams\":\"%s\"}",$0)}'`
 php $WEB_ROOT/artisan pda:update "${videoName%_noAR}" "$fieldSets"
fi
cd -

# clean up
rm -rf "$OPENPOSE_ROOT/output/$videoName"

echo
echo Completed at `date`
