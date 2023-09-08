#!/usr/bin/bash
# Usage    : For web app to submit 1 video file for processing
# Parameter: 1) Video file name (e.g AR Gait_P_H019_Free walk_09-12-2022_15-24-52_noAR.mp4)
#            2) Left Leg Length in mm
#            3) Right Leg Length in mm
#            4) Device token for push notification

videoFile="$1"
OPENPOSE_ROOT=/data/openpose

videoName="${videoFile%.mp4}"
submitName="${videoName}.sub"

# check if there is input file, exit if none
if [ ! -f "$OPENPOSE_ROOT/input/$videoFile" ]; then
 echo ERROR: $videoFile file not found in $OPENPOSE_ROOT/input
 exit 1
fi

# clear submit file older than 10 minutes to cater for unexpected residual file
find "$OPENPOSE_ROOT/input" -name "$submitName" -cmin +10 | xargs -i rm {}

# check if there is submitted file, exit if found
if [ -f "$OPENPOSE_ROOT/input/$submitName" ]; then
 echo ERROR: $submitName file found in $OPENPOSE_ROOT/input
 exit 1
fi

mv "$OPENPOSE_ROOT/input/$videoFile" "$OPENPOSE_ROOT/input/$submitName"
LOGFILE=$OPENPOSE_ROOT/log/${videoName}-`date +%Y%m%d%H%M%S`.log
echo singularity exec --nv -bind=/data $OPENPOSE_ROOT/opp20.sif $OPENPOSE_ROOT/gait/wrun.sh "$submitName" $2 $3 $4
echo $LOGFILE
singularity exec --nv --bind=/data $OPENPOSE_ROOT/opp20.sif $OPENPOSE_ROOT/gait/wrun.sh "$submitName" $2 $3 $4 >& "$LOGFILE"
#echo $!
# to avoid concurrent start up of the same Singularity container, delay is added before next submit job
# sleep 1
exit 0
