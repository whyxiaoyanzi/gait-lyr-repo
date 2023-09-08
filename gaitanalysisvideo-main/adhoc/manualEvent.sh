# Sample adhoc script run by webuser to process 1 PKL file with manual heel strike and toe off events
# Edit to change pkl file, leg length and params
# assume left and right leg length equal
FILE_NAME=ARGait_P_Uploader_Freewalk_08-08-2023_15-15-08
LEGLENGTH=880
OLEFT=0
ORIGHT=0
PARAMS=$(cat <<-END
{ "adjustOpenpose":4, "oLeft":${OLEFT}, "oRight":${ORIGHT}, 
  "LHS":[130, 221, 303, 414], 
  "RHS":[76, 180, 260, 358, 471], 
  "LTO":[86, 188, 268, 371, 483], 
  "RTO":[137, 230, 315, 425]
}
END
)

OPENPOSE_ROOT=/data/openpose
GAIT_ROOT=/data/openpose/gait
WEB_ROOT=/data/pdlogger
PKL="${WEB_ROOT}/storage/app/public/${FILE_NAME}/${FILE_NAME}_noAR.pkl"
LOGFILE=$OPENPOSE_ROOT/log/manual_${FILE_NAME}-`date +%Y%m%d%H%M%S`.log
echo singularity exec --nv -bind=/data $OPENPOSE_ROOT/opp20.sif $GAIT_ROOT/wrun_recalc.sh "$PKL" $LEGLENGTH $LEGLENGTH "$PARAMS" $OLEFT $ORIGHT
echo $LOGFILE
singularity exec --nv --bind=/data $OPENPOSE_ROOT/opp20.sif $GAIT_ROOT/wrun_recalc.sh "$PKL" $LEGLENGTH $LEGLENGTH "$PARAMS" $OLEFT $ORIGHT >& "$LOGFILE"
php $WEB_ROOT/artisan pda:update "${FILE_NAME}" "{\"assessmentRep\":4}"
exit 0