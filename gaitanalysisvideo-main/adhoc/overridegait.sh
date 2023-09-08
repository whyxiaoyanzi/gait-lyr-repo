# Sample adhoc script to process 1 PKL file with manual picked CBTA gait cycle for left and right sides
# Edit to change Openpose environment, pkl file and leg length
# assume left and right leg length equal
OPENPOSE_ROOT=/data/openpose/openpose
WEB_ROOT=/data/pdlogger
GAIT_ROOT=/data/openpose/gait
LEGLENGTH=865
PKL="/home/jslow/study1_hpc/H013/Free walk/AR Gait_P_LML_Free walk_04-11-2022_11-13-49_noAR.pkl"
OLEFT=2
ORIGHT=2

# set env variables for python scripts
export PYTHONPATH=.:$GAIT_ROOT:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/python/openpose
export MODEL_DIR=$OPENPOSE_ROOT/openpose/models
export LD_LIBRARY_PATH=$OPENPOSE_ROOT/openpose/build/python/openpose:$OPENPOSE_ROOT/openpose/build/caffe/lib:$OPENPOSE_ROOT/openpose/build/src/openpose:$LD_LIBRARY_PATH

params="{\"oLeft\":$OLEFT,\"oRight\":$ORIGHT}"
echo python3 $GAIT_ROOT/adhoc/gaitanalysis.py \"$PKL\" $LEGLENGTH $LEGLENGTH "$params"
assess_dt=${PKL#*walk_}
assess_dt=${assess_dt%_noAR*}
python3 $GAIT_ROOT/adhoc/gaitanalysis.py "$PKL" $LEGLENGTH $LEGLENGTH "$params" > run_${assess_dt}.log
grep -e ERROR -e 'full gait cycle' run_${assess_dt}.log

# gather gait analysis data
videoName=${PKL#*walk*/}
videoName=${videoName%.pkl}
cbtaName="${videoName}_cbta.csv"
savName="${videoName}_sav.csv"
gaitName="${videoName}_gait.csv"
kineName="${videoName}_kine.csv"
echo cbta > "$gaitName"
if [ -f "$cbtaName" ]; then
    awk -f $GAIT_ROOT/pickcsv.awk oLeft=$OLEFT oRight=$ORIGHT "$cbtaName" >> "$gaitName"
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
fieldSets=`awk -F',' -f $WEB_ROOT/supporting.files/update_pda.awk "$gaitName"`
php $WEB_ROOT/artisan pda:update "${videoName%_noAR}" "$fieldSets"

# generate kinematic graph data
grep -m2 -HA1 'full gait' run_${assess_dt}.log | awk -f $GAIT_ROOT/adhoc/kin1.awk oLeft=$OLEFT oRight=$ORIGHT | bash | awk -F, -f $GAIT_ROOT/adhoc/kin2.awk > kine_override.txt
echo Edit custom_gait_cycle.awk to add the following lines:
echo "/${assess_dt}.+Left/ { override(${OLEFT}) }"
echo "/${assess_dt}.+Right/ { override(${ORIGHT}) }"

rm *.png
