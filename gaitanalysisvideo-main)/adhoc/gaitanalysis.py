import gait
import os
import sys

if len(sys.argv) < 4:
    print("ERROR: Expecting pkl_file left_leg_length(mm) right_leg_length(mm) [params]")
    sys.exit(1)

pkl = sys.argv[1]
left_leg_length = int(sys.argv[2])*0.001
right_leg_length = int(sys.argv[3])*0.001
filename = pkl.split("/")[-1].split(".")[0]
imuLeft = os.path.join(os.path.dirname(pkl),filename[:-5] + "_2.csv")
imuRight = os.path.join(os.path.dirname(pkl),filename[:-5] + "_3.csv")
if len(sys.argv) == 5:
    params=sys.argv[4]
else:
    params=None

gait.analyse_all(pkl, imuLeft, imuRight, filename, left_leg_length, right_leg_length, params)