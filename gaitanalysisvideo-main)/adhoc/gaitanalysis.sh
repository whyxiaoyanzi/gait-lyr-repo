# Sample adhoc script to process PKL files
# Edit to change Openpose environment, pkl files and leg length
# assume left and right leg length equal
OPENPOSE_ROOT=/data/openpose
WEB_ROOT=/data/pdlogger
GAIT_ROOT=/data/openpose/gait
#params="{\"from-second\":1.0,\"to-second\":2.5}"

# set env variables for python scripts
export PYTHONPATH=.:$GAIT_ROOT:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/python/openpose
export MODEL_DIR=$OPENPOSE_ROOT/openpose/models
export LD_LIBRARY_PATH=$OPENPOSE_ROOT/openpose/build/python/openpose:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/src/openpose:$LD_LIBRARY_PATH

for subj in H001-880 H002-825 H003-845 H004-880 H005-810 H006-940 H007-850 H008-890 H009-815 H010-770 H011-860 H012-825 H013-865 H014-845 H015-865 H016-960 H017-875 H018-830 H019-845 H020-805 
#for subj in H001-880
do
  SUBJ=${subj%-*}
  LEGLENGTH=${subj#*-}
  for pkl in /home/jslow/study1_hpc/${SUBJ}/Free\ walk*/*pkl
  #for pkl in '/home/jslow/study1_hpc/H013/Free walk/AR Gait_P_LML_Free walk_04-11-2022_11-12-35_noAR.pkl'
  do
    echo python3 gaitanalysis.py \"$pkl\" $LEGLENGTH $LEGLENGTH "$params"
    assess_dt=${pkl#*walk_}
    assess_dt=${assess_dt%_noAR*}
    python3 gaitanalysis.py "$pkl" $LEGLENGTH $LEGLENGTH "$params" > run_${assess_dt}.log
    grep -e ERROR -e 'full gait cycle' run_${assess_dt}.log

    # gather gait analysis data
    videoName=${pkl#*walk*/}
    videoName=${videoName%.pkl}
    cbtaName="${videoName}_cbta.csv"
    savName="${videoName}_sav.csv"
    gaitName="${videoName}_gait.csv"
    kineName="${videoName}_kine.csv"
    echo cbta > "$gaitName"
    if [ -f "$cbtaName" ]; then
      awk -f $GAIT_ROOT/pickcsv.awk "$cbtaName" >> "$gaitName"
    else
      echo No result >> "$gaitName"
    fi
    echo >> "$gaitName"
    echo sav >> "$gaitName"
    if [ -f "$savName" ]; then
      awk -f $GAIT_ROOT/pickcsv.awk "$savName" >> "$gaitName"
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

    # update pd_assessment table with gait parameters
    fieldSets=`awk -F, -f $WEB_ROOT/supporting.files/update_pda.awk "$gaitName"`
    php $WEB_ROOT/artisan pda:update "${videoName%_noAR}" "$fieldSets"

  done
done

rm *.png
