GAIT_ROOT=/data/openpose/gait
echo Create list of gait cycles from run log
if [ -f exclude_gait_cycle.txt ]; then
  # sample exclude record: 09-09-2022_16-10-28
  ls -1 run*.log | grep -vf exclude_gait_cycle.txt | xargs -i grep -m2 -HA1 'full gait' {} > gait_cycle.txt
else
  ls -1 run*.log | xargs -i grep -m2 -HA1 'full gait' {} > gait_cycle.txt
fi
echo -- samples of `wc -l gait_cycle.txt` lines
head -2 gait_cycle.txt

## custom midpoint selection
if [ -f custom_gait_cycle.awk ]; then
 mv gait_cycle.txt orig_gait_cycle.txt
 awk -f custom_gait_cycle.awk orig_gait_cycle.txt > gait_cycle.txt
fi

echo Create mid cycle kinematics data
awk -f $GAIT_ROOT/adhoc/kin1.awk gait_cycle.txt | bash | awk -F, -f $GAIT_ROOT/adhoc/kin2.awk > kine_mid_t.txt

echo Create mid cycle kinematics data mapped to gait cycle for Excel scatter plot
awk -f $GAIT_ROOT/adhoc/kin1.awk gait_cycle.txt | bash | awk -F, -f $GAIT_ROOT/adhoc/kin3.awk > kine_mid_esp.txt
