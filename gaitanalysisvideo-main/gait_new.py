import gaitevents
#import openpose_video
import sys
import os
import traceback

# Openpose video processing with PKL file generated in save_dir
def process_video(walk_dir, raw_video_file, save_dir, leftLegLength, rightLegLength):
  patientinfo = {
    'leg_length_left': leftLegLength,
    'leg_length_right': rightLegLength
  }
  print(f"Processing {walk_dir} video: {raw_video_file} output to: {save_dir}")
  print(f"Leg Length (left, right) in m : ({patientinfo['leg_length_left']:.3f}, {patientinfo['leg_length_right']:.3f})")

  try:
    if walk_dir == "Side.RIGHT":
      openpose_video.process(gaitevents.Side.RIGHT, raw_video_file, save_dir, patientinfo)
    elif walk_dir == "Side.LEFT":
      openpose_video.process(gaitevents.Side.LEFT, raw_video_file, save_dir, patientinfo)
    else:
      raise ValueError(f"Unknown walk_dir: {walk_dir}.  Accept Side.RIGHT or Side.LEFT")
  except Exception as e:
    print(e)
    traceback.print_exc()
    raise

def analyse_video(pkl, params=None):
   gaitdata = gaitevents.load_object(pkl)
   if params is not None:
     gaitdata.override(params)
   return gaitevents.analyseVideo(gaitdata)

def analyse_cbta(pkl, filename, left_leg_length, right_leg_length, file_name, all_timings, params=None):
    try:
        gaitdata = gaitevents.load_object(pkl)
        gaitdata.patientinfo = {
            'leg_length_left': left_leg_length,
            'leg_length_right': right_leg_length
        }
        print("Processing CBTA for "+pkl)
        if params is not None:
            gaitdata.override(params)
        gaitevents.cbta(gaitdata,file_name,all_timings)
    except Exception:
        print("ERROR: Failed processing CBTA " + pkl)
        traceback.print_exc()
    else:
        if os.path.exists('gaitanalysis.csv') and os.path.exists('kinematics.csv'):
            os.replace('gaitanalysis.csv', filename+'_cbta.csv')
            os.replace('kinematics.csv', filename+'_kine.csv')
            os.replace('kinematics_1.csv', filename+'_kine_1.csv')
            os.replace('kinematics_2.csv', filename+'_kine_2.csv')
            os.replace('cbta_dist.csv', filename+'_cbta_1.csv')

def analyse_sav(pkl, filename, left_leg_length, right_leg_length, params=None):
    try:
        gaitdata = gaitevents.load_object(pkl)
        gaitdata.patientinfo = {
          'leg_length_left': left_leg_length,
          'leg_length_right': right_leg_length
        }
        print("Processing SAV for "+pkl)
        if params is not None:
            gaitdata.override(params)
        gaitevents.shank_ang_vel_analysis(gaitdata)
    except Exception:
        print("ERROR: Failed processing SAV " + pkl)
        traceback.print_exc()
    else:
        if os.path.exists('gaitanalysis.csv'):
            os.replace('gaitanalysis.csv', filename+'_sav.csv')

def analyse_imu(imuLeft, imuRight, filename):
    if os.path.exists(imuLeft) and os.path.exists(imuRight):
        try:
            print("Processing IMU for "+imuLeft+" "+imuRight)
            gaitevents.shank_ang_vel_imu(imuLeft, imuRight)
            os.replace('gaitanalysis.csv', filename+'_imu.csv')
        except Exception:
            print("ERROR: Failed processing IMU " + imuLeft + " " + imuRight)
            traceback.print_exc()
    else:
        print("IMU files not found: "+imuLeft+" "+imuRight)

def analyse_front(pkl, filename, params=None):
   try:
        gaitdata = gaitevents.load_object(pkl)
        print("Processing Front for "+pkl)
        if params is not None:
            gaitdata.override(params)
        gaitevents.front_analysis(gaitdata)
   except Exception:
        print("ERROR: Failed processing Front " + pkl)
        traceback.print_exc()
   else:
        if os.path.exists('gaitanalysis.csv') and os.path.exists('kinematics.csv'):
            os.replace('gaitanalysis.csv', filename+'_front.csv')
            os.replace('kinematics.csv', filename+'_kine.csv')
        if os.path.exists('kinematics_3.csv') and os.path.exists('kinematics_4.csv'):
            os.replace('kinematics_3.csv', filename+'_kine_3.csv')
            os.replace('kinematics_4.csv', filename+'_kine_4.csv')

def analyse_all(pkl, imuLeft, imuRight, filename, left_leg_length, right_leg_length, params=None):
  if analyse_video(pkl, params) == "S":
    # side view video
    analyse_cbta(pkl, filename, left_leg_length, right_leg_length, params)
    analyse_sav(pkl, filename, left_leg_length, right_leg_length, params)
  else:
    # front view video
    analyse_front(pkl, filename, params)
  analyse_imu(imuLeft, imuRight, filename)

if __name__ == '__main__':
  if len(sys.argv) < 6:
      print(f"Usage: {sys.argv[0]} walk_dir raw_video_file save_dir left_leg_length(mm) right_leg_length(mm)")
      sys.exit(1)

  walk_dir = sys.argv[1]
  raw_video_file = sys.argv[2]
  save_dir = sys.argv[3]
  leftLegLength = int(sys.argv[4])*0.001
  rightLegLength = int(sys.argv[5])*0.001
  # only main module accept leg lengths in mm, all functions expect leg lengths in m

  try:
    process_video(walk_dir, raw_video_file, save_dir, leftLegLength, rightLegLength)
  except:
    sys.exit(1)

  # Process PKL
  os.chdir(save_dir)
  video_name = os.path.splitext(os.path.basename(raw_video_file))[0] 
  pkl = video_name + ".pkl"
  imuLeft = os.path.join(os.path.dirname(raw_video_file),video_name[:-5] + "_2.csv")
  imuRight = os.path.join(os.path.dirname(raw_video_file),video_name[:-5] + "_3.csv")

  analyse_all(pkl, imuLeft, imuRight, video_name, leftLegLength, rightLegLength)

