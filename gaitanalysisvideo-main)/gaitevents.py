import numpy as np
import csv
import scipy
# import scipy.ndimage
# import scipy.signal
# import scipy.fft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import pickle
import json
import os
import copy

from enum import Enum
from enum import auto

class Joint(Enum):

    LEFT_HIP = auto()
    RIGHT_HIP = auto()
    MIDDLE_HIP = auto()

    LEFT_KNEE = auto()
    RIGHT_KNEE = auto()
    LEFT_TOES = auto() # Plural "toes" joint should ideally locate midpoint of all toes
    RIGHT_TOES = auto()
    LEFT_TOE_BIG = auto()
    RIGHT_TOE_BIG = auto()
    LEFT_TOE_SMALL = auto()
    RIGHT_TOE_SMALL = auto()
    LEFT_HEEL = auto()
    RIGHT_HEEL = auto()
    LEFT_ANKLE = auto()
    RIGHT_ANKLE = auto()

    MID_SHOULDER = auto() # or between the shoulders
    LEFT_SHOULDER = auto()
    RIGHT_SHOULDER = auto()

    NONE = auto()

class GaitEvent(Enum):

    HEELSTRIKE = auto()
    TOEOFF = auto()

class SessionInfo(Enum):

    DATE = auto()

class Side(Enum):
    LEFT = auto()
    RIGHT = auto()

    def opposite(self):
        if self == Side.LEFT:
            return Side.RIGHT
        else:
            return Side.LEFT

# Deprecated but needed to load some older .pkl files
class PatientInfo(Enum):

    NAME = auto() # string
    LEG_LENGTH = auto() # dict(Side: double)

class GaitData:

    def __init__(self, framework, fps):
        self.framework = framework
        self.fps = fps
        self.patientinfo = dict() # should be filled in by json from app
        self.walk_direction = Side.RIGHT # default walking direction to the right

        # data keys should be JointName enum members
        # Convention of data arrays: x, y, z.
        # x (horizontal axis) is always positive to the right.
        # y is always the vertical axis.
        # z is optional horizontal axis for 3D. Positive direction towards camera (right hand convention)
        self.data = dict()
        for joint in Joint:
            self.data[joint] = []
        if framework == "alphapose":
            self.confidence = dict()
            for joint in Joint:
                self.confidence[joint] = []
        if (framework == "mediapipe") or (framework == "mediapipeheavy") or (framework == "vicon") or (framework == "arkit"):
            self.data_world = dict()
            for joint in Joint:
                self.data_world[joint] = []

    def trim_frames(self, from_front, from_back):

        print("Removing "+str(from_front-1)+" frames from front and "+str(from_back)+" frames from back.")

        if from_back == 0:
            for key, value in self.data.items():
                self.data[key] = value[from_front:]
            if self.framework == "alphapose":
                for key, value in self.confidence.items():
                    self.confidence[key] = value[from_front:]
            elif self.is_world_units():
                for key, value in self.data_world.items():
                    self.data_world[key] = value[from_front:]
        else:
            for key, value in self.data.items():
                self.data[key] = value[from_front:-from_back]
            if self.framework == "alphapose":
                for key, value in self.confidence.items():
                    self.confidence[key] = value[from_front:-from_back]
            elif self.is_world_units():
                for key, value in self.data_world.items():
                    self.data_world[key] = value[from_front:-from_back]
        return self

    def trim_seconds(self, from_front, from_back):
        self.trim_frames(int(from_front*self.fps), int(from_back*self.fps))
        return self

    def keep_seconds(self, start, end):
        if self.framework == "arkit":
            for key, value in self.data_world.items():
                total_duration = len(value) / self.fps
                break
        else:
            for key, value in self.data.items():
                total_duration = len(value) / self.fps
                break
        return self.trim_seconds(start, max(total_duration-end,0))

    def trim_scale(self, start, end):
        for key, value in self.data.items():
            total_duration = len(value) / self.fps
            break
        return self.trim_seconds(start*total_duration, end*total_duration)

    def is_world_units(self):
        return self.framework == "mediapipe" or self.framework == "mediapipeheavy" or self.framework == "vicon" or self.framework == "arkit"

    # Converts parameter to the appropriate sign based on walking direction.
    def distance_param_sign_convert(self, param):
        if self.walk_direction == Side.RIGHT:
            return param
        else:
            return -param

    def handle_missing_walk_direction(self):
        # Older processed data may not have walk_direction defined
        if not hasattr(self, "walk_direction"):
            print("gaitdata walk_direction not found, assigning RIGHT")
            self.walk_direction = Side.RIGHT

    # JSLOW(20230522): add override parameters
    # oLeft, oRight: manually picked mid gait cycle (base index 1)
    # adjustOpenpose: 0 for raw, 1 for OPA, 2 for RJN
    # from-second, to-second: segments of video for analysis
    def override(self, params):
        if params:
            params_dict = json.loads(params)
            self.oLeft = params_dict.get("oLeft", 0)
            self.oRight = params_dict.get("oRight", 0)
            self.adjustOpenpose = params_dict.get("adjustOpenpose", 0)
            if "from-second" in params_dict and "to-second" in params_dict:
                self.keep_seconds(params_dict["from-second"], params_dict["to-second"])
            if "LHS" in params_dict and "RHS" in params_dict and "LTO" in params_dict and "RTO" in params_dict:
                self.manualHS = {Side.LEFT: params_dict["LHS"], Side.RIGHT: params_dict["RHS"]}
                self.manualTO = {Side.LEFT: params_dict["LTO"], Side.RIGHT: params_dict["RTO"]}
            if "videoType" in params_dict:
                self.videoType = params_dict["videoType"]

    # Attempts to fix large jumps in joint positions by assigning more accurate positions instead.
    def remedy_jump_noise(self):

        def perform_correction(data, ref_data, use_ref_data = False):
            """

            :param ref_data: data used to detect erroneuous jumps. If using a framework with world coordinates
            like mediapipe where the length scale changes with every frame, then ref_data should be set to data with regular pixel coordinates (fixed scale).
            :param data: data to be corrected.
            """

            # Coefficients used to classify noise based on movement of the ankle. These are to be in terms of fractions of the patient's shank length.
            basic_noise = 0.10 # estimate of how much the ankle joint jumps from frame to frame.
            max_movement_per_frame = 0.20 * (30/self.fps) # fps-dependent.

            jump_coeff = basic_noise + max_movement_per_frame  # if ankle moves more than this amount, a 'jump' has occurred.
            overlap_coeff = 1.5*basic_noise  # if two joint positions are within this threshold, they are determined to be at the same position, roughly.
            overlap_coeff_move = jump_coeff # taking into account motion between frames

            # Manage reuse of frames
            reuse_fatigue = 0

            # Choose axes used to calculate jumps
            end_coord_idx = 3

            for i in range(1, len(ref_data[Joint.LEFT_ANKLE])):

                reuse_fatigue_inc = False

                ank_displacement = dict()
                ank_displacement[Side.LEFT] = np.linalg.norm(
                    ref_data[Joint.LEFT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.LEFT_ANKLE][i - 1][0:end_coord_idx])
                ank_displacement[Side.RIGHT] = np.linalg.norm(
                    ref_data[Joint.RIGHT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i - 1][0:end_coord_idx])
                knee_displacement = dict()
                knee_displacement[Side.LEFT] = np.linalg.norm(
                    ref_data[Joint.LEFT_KNEE][i][0:end_coord_idx] - ref_data[Joint.LEFT_KNEE][i - 1][0:end_coord_idx])
                knee_displacement[Side.RIGHT] = np.linalg.norm(
                    ref_data[Joint.RIGHT_KNEE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_KNEE][i - 1][0:end_coord_idx])
                shank_length = dict()
                shank_length[Side.LEFT] = np.linalg.norm(ref_data[Joint.LEFT_KNEE][i - 1][0:end_coord_idx] - ref_data[Joint.LEFT_ANKLE][i - 1][0:end_coord_idx])
                shank_length[Side.RIGHT] = np.linalg.norm(ref_data[Joint.RIGHT_KNEE][i - 1][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i - 1][0:end_coord_idx])

                jump_detect = {Side.LEFT: (ank_displacement[Side.LEFT] > jump_coeff * shank_length[Side.LEFT]) or (knee_displacement[Side.LEFT] > jump_coeff * shank_length[Side.LEFT]),
                               Side.RIGHT: (ank_displacement[Side.RIGHT] > jump_coeff * shank_length[Side.RIGHT]) or (knee_displacement[Side.RIGHT] > jump_coeff * shank_length[Side.RIGHT])}

                # possible joint swap
                if jump_detect[Side.LEFT] and jump_detect[Side.RIGHT]:
                    # test if ankles are currently close to opposite ankle position in previous frame
                    ankles_swapped = (np.linalg.norm(ref_data[Joint.LEFT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i-1][0:end_coord_idx]) < overlap_coeff_move * shank_length[Side.RIGHT]) and \
                        (np.linalg.norm(ref_data[Joint.RIGHT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.LEFT_ANKLE][i-1][0:end_coord_idx]) < overlap_coeff_move * shank_length[Side.LEFT])
                    knees_swapped = (np.linalg.norm(ref_data[Joint.LEFT_KNEE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_KNEE][i-1][0:end_coord_idx]) < overlap_coeff_move * shank_length[Side.RIGHT]) and \
                        (np.linalg.norm(ref_data[Joint.RIGHT_KNEE][i][0:end_coord_idx] - ref_data[Joint.LEFT_KNEE][i-1][0:end_coord_idx]) < overlap_coeff_move * shank_length[Side.LEFT])

                    if ankles_swapped or knees_swapped:
                        print("swap fix at frame " + str(i) + ". left shank length: " + str(
                            shank_length[Side.LEFT]) + ". left ank disp: " + str(
                            ank_displacement[Side.LEFT]) + ". right ank disp: " + str(ank_displacement[Side.RIGHT]))
                        for k in range(len(important_joints[Side.LEFT])):
                            temp_left = copy.deepcopy(data[important_joints[Side.LEFT][k]][i])
                            data[important_joints[Side.LEFT][k]][i] = copy.deepcopy(
                                data[important_joints[Side.RIGHT][k]][i][:])
                            data[important_joints[Side.RIGHT][k]][i] = temp_left
                            if use_ref_data:
                                temp_left = copy.deepcopy(ref_data[important_joints[Side.LEFT][k]][i])
                                ref_data[important_joints[Side.LEFT][k]][i] = copy.deepcopy(
                                    ref_data[important_joints[Side.RIGHT][k]][i][:])
                                ref_data[important_joints[Side.RIGHT][k]][i] = temp_left
                        print("new left ank: " + str(data[important_joints[Side.LEFT][k]][i]))
                        print("new right ank: " + str(data[important_joints[Side.RIGHT][k]][i]))

                    else:
                        print("Detected jumps in both ankles, but skipping swap fix at frame "+str(i)+". Replacing with values from previous frame. Consecutive reuse count: "+str(reuse_fatigue))

                        for k in range(len(important_joints[Side.LEFT])):
                            for side in Side:
                                data[important_joints[side][k]][i] = reuse_weight(reuse_fatigue) * data[important_joints[side][k]][i - 1] + (1-reuse_weight(reuse_fatigue)) * data[important_joints[side][k]][i]  # replace with position from previous timestamp
                                if use_ref_data:
                                    ref_data[important_joints[side][k]][i] = reuse_weight(reuse_fatigue) * ref_data[important_joints[side][k]][i - 1] + (1-reuse_weight(reuse_fatigue)) * ref_data[important_joints[side][k]][i] # replace with position from previous timestamp

                        reuse_fatigue += 1
                        reuse_fatigue_inc = True

                # possible overlap jump (one foot jumps to the other while the other foot remains in place)
                elif jump_detect[Side.LEFT] or jump_detect[Side.RIGHT]:

                    if jump_detect[Side.LEFT]:
                        jumped_side = Side.LEFT
                        jumped_ankle = Joint.LEFT_ANKLE
                    else:
                        jumped_side = Side.RIGHT
                        jumped_ankle = Joint.RIGHT_ANKLE

                    ankles_overlapped = np.linalg.norm(ref_data[Joint.LEFT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i][0:end_coord_idx]) > overlap_coeff * shank_length[jumped_side]
                    knees_overlapped = np.linalg.norm(ref_data[Joint.LEFT_KNEE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_KNEE][i][0:end_coord_idx]) > overlap_coeff * shank_length[jumped_side]

                    if ankles_overlapped or knees_overlapped:
                        print("overlap fix skipped at frame " + str(i) + ": " + jumped_side.name + ". Ankles not close enough.")
                        print("shank length: " + str(shank_length[jumped_side]))
                        print("separation: " + str(np.linalg.norm(ref_data[Joint.LEFT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i][0:end_coord_idx])))
                        if np.linalg.norm(ref_data[jumped_ankle][i][0:end_coord_idx] - ref_data[jumped_ankle][i-1][0:end_coord_idx]) > 2*jump_coeff * shank_length[jumped_side]:
                            print("Replacing jumped ankle with previous frame data due to significantly large jump. Consecutive reuse count: "+str(reuse_fatigue))
                            for k in range(len(important_joints[jumped_side])):
                                data[important_joints[jumped_side][k]][i] = reuse_weight(reuse_fatigue) * data[important_joints[jumped_side][k]][i - 1] + (1-reuse_weight(reuse_fatigue)) * data[important_joints[jumped_side][k]][i] # replace with position from previous timestamp
                                if use_ref_data:
                                    ref_data[important_joints[jumped_side][k]][i] = reuse_weight(reuse_fatigue) * ref_data[important_joints[jumped_side][k]][i - 1]  + (1-reuse_weight(reuse_fatigue)) * ref_data[important_joints[jumped_side][k]][i] # replace with position from previous timestamp

                            reuse_fatigue += 1
                            reuse_fatigue_inc = True

                    else:
                        print("performing overlap fix at frame " + str(i) + ": " + jumped_side.name)
                        print("shank length: " + str(shank_length[jumped_side]))
                        print("separation: " + str(np.linalg.norm(ref_data[Joint.LEFT_ANKLE][i][0:end_coord_idx] - ref_data[Joint.RIGHT_ANKLE][i][0:end_coord_idx])))
                        for k in range(len(important_joints[jumped_side])):
                            data[important_joints[jumped_side][k]][i] = data[important_joints[jumped_side][k]][i - 1]  # replace with position from previous timestamp
                            if use_ref_data:
                                ref_data[important_joints[jumped_side][k]][i] = ref_data[important_joints[jumped_side][k]][i - 1]  # replace with position from previous timestamp

                if not reuse_fatigue_inc:
                    reuse_fatigue = 0

        def reuse_weight(fatigue):
            if fatigue == 0:
                return 1
            elif fatigue == 1:
                return 0.6
            elif fatigue == 2:
                return 0.2
            elif fatigue > 2:
                return 0

        print("Attempting to remedy jumps in joints...")

        important_joints = dict() # arrays of both sides must have the same order of joints
        if self.framework == "mediapipe" or "mediapipeheavy":
            important_joints[Side.LEFT]= [Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, Joint.LEFT_HEEL, Joint.LEFT_TOE_BIG]
            important_joints[Side.RIGHT] = [Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, Joint.RIGHT_HEEL, Joint.RIGHT_TOE_BIG]
        elif self.framework == "arkit":
            important_joints[Side.LEFT]= [Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, Joint.LEFT_TOE_BIG]
            important_joints[Side.RIGHT] = [Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, Joint.RIGHT_TOE_BIG]
        else:
            important_joints[Side.LEFT]= [Joint.LEFT_HIP, Joint.LEFT_KNEE, Joint.LEFT_ANKLE, Joint.LEFT_HEEL, Joint.LEFT_TOE_BIG, Joint.LEFT_TOE_SMALL]
            important_joints[Side.RIGHT] = [Joint.RIGHT_HIP, Joint.RIGHT_KNEE, Joint.RIGHT_ANKLE, Joint.RIGHT_HEEL, Joint.RIGHT_TOE_BIG, Joint.RIGHT_TOE_SMALL]
        if len(important_joints[Side.LEFT]) != len(important_joints[Side.RIGHT]):
            raise ValueError("Error when remedying joint position jumps: different number of joints to be corrected specified for each side.")

        if self.framework == "arkit":
            perform_correction(self.data_world, self.data_world)
        elif self.is_world_units():
            perform_correction(self.data_world, self.data, True)
        else:
            perform_correction(self.data, self.data)

        print("Jump remedy end.\n")
        return self

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, protocol=4)

def load_object(filename):
    with open(filename, 'rb') as input:
        return pickle.load(input)

def alphapose_to_gaitdata(walk_direction: Side, main_dir, fps, patientinfo = dict()):
    # Assume Json data format is default (COCO)
    # Assume 26 HALPE keypoints.
    # From HALPE GitHub: https://github.com/Fang-Haoshu/Halpe-FullBody
    #     //26 body keypoints
    #     {0,  "Nose"},
    #     {1,  "LEye"},
    #     {2,  "REye"},
    #     {3,  "LEar"},
    #     {4,  "REar"},
    #     {5,  "LShoulder"},
    #     {6,  "RShoulder"},
    #     {7,  "LElbow"},
    #     {8,  "RElbow"},
    #     {9,  "LWrist"},
    #     {10, "RWrist"},
    #     {11, "LHip"},
    #     {12, "RHip"},
    #     {13, "LKnee"},
    #     {14, "Rknee"},
    #     {15, "LAnkle"},
    #     {16, "RAnkle"},
    #     {17,  "Head"},
    #     {18,  "Neck"},
    #     {19,  "Hip"},
    #     {20, "LBigToe"},
    #     {21, "RBigToe"},
    #     {22, "LSmallToe"},
    #     {23, "RSmallToe"},
    #     {24, "LHeel"},
    #     {25, "RHeel"},
    
    def find_frame_id(image_id: str):
        return int(image_id.split(".")[0])

    load_file = main_dir+"/alphapose-results.json"
    save_dir = main_dir

    json_array = json.load(open(load_file))
    keypoint_num = 26

    # keypoints_all: keys are keypoint ids, values each contain an n by 3 array where n is number of frames.
    keypoints_all = dict()
    for kp in range(keypoint_num):
        keypoints_all[kp] = []

    prev_img_id = ""
    for json_dict in json_array:

        if prev_img_id != "":
            missing_count = find_frame_id(json_dict["image_id"]) - (find_frame_id(prev_img_id) + 1)
        else:
            missing_count = 0

        if json_dict["image_id"] == prev_img_id:
            # Skip other detected "persons" in the same frame
            continue
        elif missing_count > 0:
            # if a gap in frames is encountered (which happens when no persons are detected)
            print("No persons detected in " + str(missing_count) + " frames, filling in from previous frame...")
            for i in range(missing_count):
                # fill in all missing frames
                for kp in range(keypoint_num):
                    frame_entry = keypoints_all[kp][-1][:]
                    keypoints_all[kp].append(frame_entry)

        prev_img_id = json_dict["image_id"]
        for kp in range(keypoint_num):
            frame_entry = json_dict["keypoints"][kp*3:(kp*3)+3]
            frame_entry[1] = -frame_entry[1] # flip y-axis
            keypoints_all[kp].append(frame_entry)
    #print(keypoints_all)

    gaitdata = GaitData("alphapose", fps)
    gaitdata.walk_direction = walk_direction
    gaitdata.patientinfo = patientinfo

    gaitdata.data[Joint.LEFT_HEEL] = np.array(keypoints_all[24])[:, 0:2]
    gaitdata.data[Joint.RIGHT_HEEL] = np.array(keypoints_all[25])[:, 0:2]
    gaitdata.data[Joint.LEFT_TOE_SMALL] = np.array(keypoints_all[22])[:, 0:2]
    gaitdata.data[Joint.RIGHT_TOE_SMALL] = np.array(keypoints_all[23])[:, 0:2]
    gaitdata.data[Joint.LEFT_TOE_BIG] = np.array(keypoints_all[20])[:, 0:2]
    gaitdata.data[Joint.RIGHT_TOE_BIG] = np.array(keypoints_all[21])[:, 0:2]
    gaitdata.data[Joint.LEFT_ANKLE] = np.array(keypoints_all[15])[:, 0:2]
    gaitdata.data[Joint.RIGHT_ANKLE] = np.array(keypoints_all[16])[:, 0:2]
    gaitdata.data[Joint.LEFT_KNEE] = np.array(keypoints_all[13])[:, 0:2]
    gaitdata.data[Joint.RIGHT_KNEE] = np.array(keypoints_all[14])[:, 0:2]
    gaitdata.data[Joint.LEFT_HIP] = np.array(keypoints_all[11])[:, 0:2]
    gaitdata.data[Joint.RIGHT_HIP] = np.array(keypoints_all[12])[:, 0:2]
    gaitdata.data[Joint.MIDDLE_HIP] = np.array(keypoints_all[19])[:, 0:2]

    mid_shoulder = np.array(keypoints_all[5])[:,0:2] + 0.5 * (np.array(keypoints_all[6])[:,0:2] - np.array(keypoints_all[5])[:,0:2])
    gaitdata.data[Joint.MID_SHOULDER] = mid_shoulder

    mid_left_toes = np.array(keypoints_all[20])[:,0:2] + 0.5 * (np.array(keypoints_all[22])[:,0:2] - np.array(keypoints_all[20])[:,0:2])
    mid_right_toes = np.array(keypoints_all[21])[:, 0:2] + 0.5 * (np.array(keypoints_all[23])[:, 0:2] - np.array(keypoints_all[21])[:, 0:2])
    gaitdata.data[Joint.LEFT_TOES] = mid_left_toes
    gaitdata.data[Joint.RIGHT_TOES] = mid_right_toes

    save_object(gaitdata, save_dir+"/gaitdata_alphapose.pkl")

    print("gaitdata saved as "+save_dir+"/gaitdata_alphapose.pkl")

def vicon_to_gaitdata(vicon_csv: str, save_dir: str):

    vicon_freq = 100 # 100hz

    joint_col = {}
    joint_col[Joint.LEFT_HIP] = 2
    joint_col[Joint.RIGHT_HIP] = 5
    joint_col[Joint.LEFT_KNEE] = 17
    joint_col[Joint.LEFT_ANKLE] = 23
    joint_col[Joint.LEFT_HEEL] = 26
    joint_col[Joint.LEFT_TOE_BIG] = 29
    joint_col[Joint.LEFT_TOES] = 29
    joint_col[Joint.RIGHT_KNEE] = 35
    joint_col[Joint.RIGHT_ANKLE] = 41
    joint_col[Joint.RIGHT_HEEL] = 44
    joint_col[Joint.RIGHT_TOE_BIG] = 47
    joint_col[Joint.RIGHT_TOES] = 47

    gaitdata = GaitData("vicon", vicon_freq)

    csvfile = open(vicon_csv, newline='')
    csvreader = csv.reader(csvfile)

    traj_count = -1
    frame_current = 1
    true_start_threshold = 15 # in frames
    start_confirmed = False
    end_strike = 0
    end_strike_threshold = 10

    for row in csvreader:
        if len(row) == 0:
            continue
        if row[0] == "Trajectories":
            traj_count = 0
        elif traj_count > -1 and traj_count < 5:
            traj_count += 1
        if traj_count >= 5: # start of coordinate data rows
            if traj_count == 5:
                frame_start = int(row[0]) # absolute start of data rows in csv
                start = frame_start # start of gaitdata
                traj_count += 1
            frame_current = int(row[0])

            valid_row = True
            for col in joint_col.values():
                if len(row)-1 < col or not row[col]: # check for empty entry
                    valid_row = False
                    break
            if valid_row:
                end_strike = 0
                if not start_confirmed:
                    # print(frame_current)
                    # print(start)
                    if start == frame_start:
                        start = frame_current
                    if start > frame_start and frame_current - start > true_start_threshold:
                        start_confirmed = True
                        print("start confirmed")
                    else:
                        print(row)
                        for joint, col in joint_col.items():
                            gaitdata.data_world[joint].append([float(row[col]), float(row[col + 2]), float(row[col + 1])])

            if not valid_row:
                if not start_confirmed and start != frame_start:
                    start = frame_start
                    gaitdata = GaitData("vicon", vicon_freq) # reset gaitdata, since invalid data found before threshold
                    print("reset")
                elif start_confirmed:
                    print("end strike: "+str(end_strike))
                    end_strike += 1
                    if end_strike > end_strike_threshold:
                        for joint, col in joint_col.items():
                            gaitdata.data_world[joint] = gaitdata.data_world[joint][0:-end_strike]
                        break # end of data detected

            if start_confirmed:
                if not valid_row:
                    print("Some entries missing. Using previous row values to fill in.")
                print(row)
                for joint, col in joint_col.items():
                    entries = []
                    for i in [0, 2, 1]:
                        if not row[col+i]:
                            entries.append(gaitdata.data_world[joint][-1][i])
                        else:
                            entries.append(float(row[col+i]))
                    gaitdata.data_world[joint].append(entries)

    for joint in joint_col.keys():
        gaitdata.data_world[joint] = np.array(gaitdata.data_world[joint])

    mid_left_toes = gaitdata.data_world[Joint.LEFT_TOE_BIG] + 0.5 * (
                gaitdata.data_world[Joint.LEFT_TOE_SMALL] - gaitdata.data_world[Joint.LEFT_TOE_BIG])
    mid_right_toes = gaitdata.data_world[Joint.RIGHT_TOE_BIG] + 0.5 * (
            gaitdata.data_world[Joint.RIGHT_TOE_SMALL] - gaitdata.data_world[Joint.RIGHT_TOE_BIG])
    gaitdata.data_world[Joint.LEFT_TOES] = mid_left_toes
    gaitdata.data_world[Joint.RIGHT_TOES] = mid_right_toes

    gaitdata.vicon_start = start # marks down the vicon frame at which gaitdata starts
    save_object(gaitdata, save_dir+"/gaitdata_vicon.pkl")

# Assumes subject is walking diagonally towards camera and to the right.
# At the moment, takes negative z axis as primary axis (temporary and will need to be changed accordingly)
def ar_csv_to_gaitdata(walk_direction: Side, ar_csv_file: str, save_dir: str):

    def header_to_joint(header: str):

        if header.split("_")[-1] == "x":
            removed_x = "_".join(header.split("_")[:-1])
        else:
            removed_x = header

        if removed_x == "hips_joint":
            return Joint.MIDDLE_HIP
        elif removed_x == "left_upLeg_joint":
            return Joint.LEFT_HIP
        elif removed_x == "right_upLeg_joint":
            return Joint.RIGHT_HIP
        elif removed_x == "left_leg_joint":
            return Joint.LEFT_KNEE
        elif removed_x == "right_leg_joint":
            return Joint.RIGHT_KNEE
        elif removed_x == "left_foot_joint":
            return Joint.LEFT_ANKLE
        elif removed_x == "right_foot_joint":
            return Joint.RIGHT_ANKLE
        elif removed_x == "left_toesEnd_joint":
            return Joint.LEFT_TOE_BIG
        elif removed_x == "right_toesEnd_joint":
            return Joint.RIGHT_TOE_BIG
        elif removed_x == "left_shoulder_1_joint":
            return Joint.LEFT_SHOULDER
        elif removed_x == "right_shoulder_1_joint":
            return Joint.RIGHT_SHOULDER
        elif removed_x == "neck_1_joint":
            return Joint.MID_SHOULDER
        else:
            return Joint.NONE

    gaitdata = GaitData("arkit", 60)
    gaitdata.walk_direction = walk_direction

    joint_header_idx = dict()

    row_num = 0
    with open(ar_csv_file, newline='', errors='ignore') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:

            row_num += 1
            # header row
            if row_num == 1:
                for i, header in enumerate(row):
                    if header_to_joint(header) != Joint.NONE:
                        joint_header_idx[i] = header_to_joint(header)

            else:
                for idx in joint_header_idx:
                    gaitdata.data_world[joint_header_idx[idx]].append([-float(row[idx+2]), float(row[idx+1]), float(row[idx])])

    # Use ankles for heels, because arkit does not give heel data
    gaitdata.data_world[Joint.LEFT_HEEL] = gaitdata.data_world[Joint.LEFT_ANKLE]
    gaitdata.data_world[Joint.RIGHT_HEEL] = gaitdata.data_world[Joint.RIGHT_ANKLE]

    for joint in gaitdata.data_world.keys():
        gaitdata.data_world[joint] = np.array(gaitdata.data_world[joint])
    save_object(gaitdata, save_dir + "/gaitdata_arkit.pkl")
    print("AR gaitdata saved in "+ save_dir + "/gaitdata_arkit.pkl")

def simple_chart(raw_data, fps):

    # Assuming forwards is positive

    #sos = scipy.signal.butter(4, 3.0, 'lp', fs=fps, output='sos')
    #data_smoothed = scipy.signal.sosfilt(sos, raw_data)
    #data_smoothed, _ = scipy.signal.sosfilt(sos, raw_data, zi=scipy.signal.sosfilt_zi(sos))

    b, a = scipy.signal.butter(2, 3.0, 'lp', fs=fps, output='ba')
    data_smoothed = scipy.signal.filtfilt(b, a, raw_data, padlen=10)

    #data_smoothed = simple_moving_average(raw_data, 9)

    #data_smoothed = exp_moving_average(raw_data, 0.10)

    #data_smoothed = scipy.signal.medfilt(raw_data, kernel_size=7)

    #data_smoothed = scipy.ndimage.gaussian_filter1d(raw_data, 3.0)

    time_axis = np.linspace(0, len(data_smoothed)/fps, num=len(data_smoothed))

    fig_l = plt.figure("Simple chart")
    plt.plot(time_axis, raw_data)
    plt.plot(time_axis, data_smoothed)
    plt.xlabel("Time (seconds)")
    plt.ylabel("X-coordinate of left heel")

# def plot_knee_flexion(gaitdata):
#     if (gaitdata.framework == "mediapipe") or (gaitdata.framework == "mediapipeheavy") or (gaitdata.framework == "vicon"):
#         data = gaitdata.data_world
#     else:
#         data = gaitdata.data
#     flexion_lknee = []
#     for i in range(len(data[Joint.LEFT_KNEE])):
#         flexion_lknee.append(find_flexion_deg( data[Joint.LEFT_HIP][i], data[Joint.LEFT_KNEE][i], data[Joint.LEFT_ANKLE][i]))
#     plt.plot(flexion_lknee, '--x')

def plot_joint(gaitdata, joint):
    data = gaitdata.data_world
    time_axis = np.linspace(0, len(data[joint][:, 0])/gaitdata.fps, num=len(data[joint][:, 0]))
    x, = plt.plot(time_axis, data[joint][:, 0], label='X-coordinate')
    y, = plt.plot(time_axis, data[joint][:, 1], label='Y-coordinate')
    plt.legend(handles=[x, y])
    plt.xlabel("Time (seconds)")
    plt.ylabel("Coordinate (pixels)")

#JSLOW(20230630): modify heel and toe raw (x,y) translated to middle hip point 
#JSLOW(20230807): extend to other keypoints but first 2 in the list are heel and toe raw
def openposeAdjustment(keypoints, fps):
    kpNames = ['heel', 'toe', 'ankle', 'knee', 'hip']
    # threshold in pixel for significance of difference
    delta_threshold = 25
    #JSLOW(20230703) - handle heel and toe far apart
    footToLegRatio = {
        Side.LEFT: np.linalg.norm(keypoints[0][Side.LEFT][:]-keypoints[1][Side.LEFT][:], axis=1) / np.linalg.norm(keypoints[0][Side.LEFT][:], axis=1),
        Side.RIGHT: np.linalg.norm(keypoints[0][Side.RIGHT][:]-keypoints[1][Side.RIGHT][:], axis=1) / np.linalg.norm(keypoints[0][Side.RIGHT][:], axis=1)
    }
    # dictionary of rejection reason map to a set of frame no
    rejected_frames = {}
    b, a = scipy.signal.butter(N=3, Wn=6.0, btype='lowpass', fs=fps, output='ba')
    for side in Side:
        if len(keypoints)==2:
            # only check big foot small foot conditions for heel and toe adjustments
            # threshold to identify large distance between heel and toe
            footToLegRatio_threshold = 2 * np.mean(footToLegRatio[side])
            rejected_frames['big foot'] = {i for i,v in enumerate(footToLegRatio[side]) if v > footToLegRatio_threshold}
            # only correct small foot that occurs in exactly 1 frame
            rejected_frames['small foot'] = {i for i,v in enumerate(footToLegRatio[side]) if v < footToLegRatio_threshold/8}.difference(
                {i-1 for i,v in enumerate(footToLegRatio[side]) if v < footToLegRatio_threshold/8},
                {i+1 for i,v in enumerate(footToLegRatio[side]) if v < footToLegRatio_threshold/8}
                )

        #JSLOW(20230713) - if x or y changes too drastic, re-estimate bad frames
        for k,kp in enumerate(keypoints):
            x_noise = kp[side][:,0] - scipy.signal.filtfilt(b, a, kp[side][:,0], axis=0, padlen=20)
            y_noise = kp[side][:,1] - scipy.signal.filtfilt(b, a, kp[side][:,1], axis=0, padlen=20)
            rejected_frames[kpNames[k]+' noise'] = {i for (i,noise) in enumerate(x_noise) if abs(noise) > 2*delta_threshold}.union(
                                        {i for (i,noise) in enumerate(y_noise) if abs(noise) > 2*delta_threshold})
       
        all_rejected_frames = set().union(*rejected_frames.values())
        keep_index = [i for i,_ in enumerate(footToLegRatio[side]) if i not in all_rejected_frames]
        fx, fy = {}, {}
        for k,kp in enumerate(keypoints):
            fx[kpNames[k]] = scipy.interpolate.CubicSpline(keep_index,kp[side][keep_index,0],bc_type='natural')
            fy[kpNames[k]] = scipy.interpolate.CubicSpline(keep_index,kp[side][keep_index,1],bc_type='natural')

        for i in sorted(all_rejected_frames):
            reasons = { reason for reason,frames in rejected_frames.items() if i in frames }
            if not reasons.isdisjoint({'big foot','small foot'}):
                beforeChg=f"({keypoints[0][side][i,0]:.2f},{keypoints[0][side][i,1]:.2f})-({keypoints[1][side][i,0]:.2f},{keypoints[1][side][i,1]:.2f})"
                keypoints[0][side][i,0], keypoints[0][side][i,1] = fx['heel'](i), fy['heel'](i)
                keypoints[1][side][i,0], keypoints[1][side][i,1] = fx['toe'](i), fy['toe'](i)
                afterChg=f"({keypoints[0][side][i,0]:.2f},{keypoints[0][side][i,1]:.2f})-({keypoints[1][side][i,0]:.2f},{keypoints[1][side][i,1]:.2f})"
                print(f"OPA:{side.name} update heel-toe at frame {i} {beforeChg} to {afterChg} due to {', '.join(reasons)}")
            
            for k,kp in enumerate(keypoints):
                if not reasons.isdisjoint({kpNames[k]+' noise'}):
                    beforeChg=f"({kp[side][i,0]:.2f},{kp[side][i,1]:.2f})"
                    kp[side][i,0], kp[side][i,1] = fx[kpNames[k]](i), fy[kpNames[k]](i)
                    afterChg=f"({kp[side][i,0]:.2f},{kp[side][i,1]:.2f})"
                    print(f"OPA:{side.name} update {kpNames[k]} at frame {i} {beforeChg} to {afterChg} due to {', '.join(reasons)}")
    # return a dictionary based on input params
    retValue = {}
    for k,kp in enumerate(keypoints):
        retValue[kpNames[k]] = kp
    return retValue

# Analyse type of video
# return S : side, F : front
def analyseVideo(gaitdata):
    # if run parameter override video type, return the value (S/F)
    if hasattr(gaitdata,"videoType"):
        print(f"analyseVideo: Manual video type {gaitdata.videoType}")
        return gaitdata.videoType
    
    data = gaitdata.data
    # translate to middle hip coordinate as origin
    knee_raw = {Side.LEFT: np.array(data[Joint.LEFT_KNEE] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_KNEE] - data[Joint.MIDDLE_HIP])}
    hip_raw = {Side.LEFT: np.array(data[Joint.LEFT_HIP] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_HIP] - data[Joint.MIDDLE_HIP])}
    
    videoType = "S"  # default assume side video
    # if knee-x is always same sign, indicates frontal video
    # if left hip-x > 0, indicates facing camera else back facing camera 
    frameCount = len(knee_raw[Side.LEFT][:,0])
    samplesA = np.arange(int(0.10*frameCount), int(0.11*frameCount))
    samplesB = frameCount - samplesA
    leftOnRightSideCount = np.sum(np.sign(knee_raw[Side.LEFT][:,0])+1)/2
    print(f"DEBUG: {leftOnRightSideCount/frameCount}")              
    if leftOnRightSideCount/frameCount > 0.9:
        print(f"analyseVideo: Front facing view [{leftOnRightSideCount}/{frameCount}] frames with left knee on right side")
        videoType = "F"
    elif np.min(hip_raw[Side.LEFT][samplesA,0]) > 0 and np.min(hip_raw[Side.LEFT][samplesB,0]) < 0:
        if np.min(hip_raw[Side.LEFT][samplesA,1] - hip_raw[Side.RIGHT][samplesA,1]) > 0:
            print("analyseVideo: Walk to Right direction with stationary camera in the middle")
            if gaitdata.walk_direction != Side.RIGHT:
                print("analyseVideo: [WARN] Inconsistent with gaitdata walk direction")
        else:
            print("analyseVideo: Walk to Left direction with stationary camera in the middle")
            if gaitdata.walk_direction != Side.LEFT:
                print("analyseVideo: [WARN] Inconsistent with gaitdata walk direction")
    else:
        print("analyseVideo: [WARN] Unknown walk pattern, fallback to default side video")
    return videoType

def front_analysis(gaitdata):
    
    data = gaitdata.data
    # translate to middle hip coordinate as origin
    ankle_raw = {Side.LEFT: np.array(data[Joint.LEFT_ANKLE] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_ANKLE] - data[Joint.MIDDLE_HIP])}
    knee_raw = {Side.LEFT: np.array(data[Joint.LEFT_KNEE] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_KNEE] - data[Joint.MIDDLE_HIP])}
    hip_raw = {Side.LEFT: np.array(data[Joint.LEFT_HIP] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_HIP] - data[Joint.MIDDLE_HIP])}
    heel_raw = {Side.LEFT: np.array(data[Joint.LEFT_HEEL] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_HEEL] - data[Joint.MIDDLE_HIP])}
    toe_raw = {Side.LEFT: np.array(data[Joint.LEFT_TOE_BIG] - data[Joint.MIDDLE_HIP]), 
                Side.RIGHT: np.array(data[Joint.RIGHT_TOE_BIG] - data[Joint.MIDDLE_HIP])}
    
    #JSLOW(20230630) - adjust keypoints detected by Openpose
    if hasattr(gaitdata,"adjustOpenpose"):
        if gaitdata.adjustOpenpose == 1:
            adjustedKeypoints = openposeAdjustment([heel_raw, toe_raw, ankle_raw, knee_raw, hip_raw], gaitdata.fps)
            heel_raw, toe_raw = adjustedKeypoints['heel'], adjustedKeypoints['toe']
            ankle_raw, knee_raw, hip_raw = adjustedKeypoints['ankle'], adjustedKeypoints['knee'], adjustedKeypoints['hip']

    padlength = 5 
    b, a = scipy.signal.butter(3, 7.0, 'lp', fs=gaitdata.fps, output='ba')
    ankle_filt, knee_filt, hip_filt, heel_filt, toe_filt = {}, {}, {}, {}, {}
    for side in Side:
        ankle_filt[side] = scipy.signal.filtfilt(b, a, ankle_raw[side], axis=0, padlen=padlength)
        knee_filt[side] = scipy.signal.filtfilt(b, a, knee_raw[side], axis=0, padlen=padlength)
        hip_filt[side] = scipy.signal.filtfilt(b, a, hip_raw[side], axis=0, padlen=padlength)
        heel_filt[side] = scipy.signal.filtfilt(b, a, heel_raw[side], axis=0, padlen=padlength)
        toe_filt[side] = scipy.signal.filtfilt(b, a, toe_raw[side], axis=0, padlen=padlength)

    angle_rad = np.arctan((hip_filt[Side.LEFT][:,1]-hip_filt[Side.RIGHT][:,1])/(hip_filt[Side.LEFT][:,0]-hip_filt[Side.RIGHT][:,0]))
    pelvic_obliquity = {Side.LEFT:np.rad2deg(angle_rad), Side.RIGHT:-np.rad2deg(angle_rad)}

    csvkinetime = {Side.LEFT: open('kinematics_3.csv','w', newline=''),
                   Side.RIGHT: open('kinematics_4.csv','w', newline='')}
    cktwriter = {Side.LEFT: csv.writer(csvkinetime[Side.LEFT], lineterminator='\n'), 
                 Side.RIGHT: csv.writer(csvkinetime[Side.RIGHT], lineterminator='\n')}
    hip_adduction, knee_adduction, heel_distance = {}, {}, {}
    for side in Side:
        time_axis = np.linspace(0, len(angle_rad) / gaitdata.fps, num=len(angle_rad))
        hip_adduction[side], knee_adduction[side] = [], []
        for i in range(len(time_axis)):
            hip_adduction[side].append(-find_flexion_deg(side, knee_filt[side][i], hip_filt[side][i], knee_filt[side][i], [knee_filt[side][i,0],0]))
            knee_adduction[side].append(-find_flexion_deg(side, knee_filt[side][i], hip_filt[side][i], knee_filt[side][i], ankle_filt[side][i]))
        # heel_move_distance = np.append([0], np.linalg.norm(heel_filt[side][1:]-heel_filt[side][:-1], axis=1))
        # toe_move_distance = np.append([0], np.linalg.norm(toe_filt[side][1:]-toe_filt[side][:-1], axis=1))
        # heel_move_distance = np.linalg.norm(heel_filt[side], axis=1) / np.linalg.norm(hip_filt[Side.LEFT]-hip_filt[Side.RIGHT], axis=1)
        heel_distance[side] = -heel_filt[side][:,1] / np.linalg.norm(hip_filt[Side.LEFT]-hip_filt[Side.RIGHT], axis=1)
        # toe_move_distance = np.linalg.norm(heel_filt[side]-toe_filt[side], axis=1) / np.linalg.norm(hip_filt[Side.LEFT]-hip_filt[Side.RIGHT], axis=1)
        # footsize_change = np.append([0], (toe_move_distance[1:]-toe_move_distance[:-1]))
        # footsize_change = -toe_filt[side][:,1] / np.linalg.norm(hip_filt[Side.LEFT]-hip_filt[Side.RIGHT], axis=1)
        cktwriter[side].writerow(["time","pelvic_obliquity","hip_adduction", "knee_adduction", "heel_distance"])
        frontKine = np.array([time_axis, pelvic_obliquity[side], hip_adduction[side], knee_adduction[side], heel_distance[side]]).T
        cktwriter[side].writerows(frontKine)
        hip_adduction[side] = np.array(hip_adduction[side])
        knee_adduction[side] = np.array(knee_adduction[side])

    kinedata = {
        'pelvic obliquity': {Side.LEFT:pelvic_obliquity[Side.LEFT], Side.RIGHT:pelvic_obliquity[Side.RIGHT]},
        'hip adduction': {Side.LEFT:hip_adduction[Side.LEFT], Side.RIGHT:hip_adduction[Side.RIGHT]},
        'knee adduction': {Side.LEFT:knee_adduction[Side.LEFT], Side.RIGHT:knee_adduction[Side.RIGHT]}
    }
    #JSLOW(20230807) - manually selected HS TO frames
    if hasattr(gaitdata,"manualHS") and hasattr(gaitdata,"manualTO"):
        print("Front HSTO bypassed using manually selected HS TO events")
        print_analysis_HSTO(gaitdata.manualHS, gaitdata.manualTO, use_gaitdata=False, gaitdata=gaitdata, fps=gaitdata.fps, kinedata=kinedata)
        return
    
    # estimate TO events from peak knee_adduction
    f, psd, stride_freq = {}, {}, {}
    min_separation_coeff = 0.75  # Used to decide minimum duration between same event over different cycles
    for side in Side:
        f[side], psd[side] = scipy.signal.periodogram(knee_adduction[side], fs=gaitdata.fps)
        stride_freq[side] = f[side][np.argmax(psd[side])]
        print(side.name+" detected stride frequency: "+str(stride_freq[side])+"Hz")

    #JSLOW(20230714) - pick higher detected stride frequency if one side is too low
    if stride_freq[Side.LEFT]<0.1:
        stride_freq[Side.LEFT] = max(stride_freq[Side.LEFT], stride_freq[Side.RIGHT])
    if stride_freq[Side.RIGHT]<0.1:
        stride_freq[Side.RIGHT] = max(stride_freq[Side.LEFT], stride_freq[Side.RIGHT])

    HS, TO, events = {}, {}, {}
    for side in Side: 
        TO[side], _ = scipy.signal.find_peaks(-knee_adduction[side], distance=((1/stride_freq[side]) * min_separation_coeff) * gaitdata.fps)
        TO[side] = TO[side] - int(0.075*gaitdata.fps)  # estimate 0.075s before mid swing is toe off event
        for frame in TO[side]:
            events[frame] = (side, "TO")

    #check if have at least 4 alternating TO sequence for HS detection
    TO_timings = sorted(events.keys())
    alternatingSequence = True
    if len(TO_timings) >= 4: 
        for i in range(len(TO_timings)-1):
            # check if consecutive TO events are on the same side
            if events[TO_timings[i]][0] == events[TO_timings[i+1]][0]:
                alternatingSequence = False
                break
        if alternatingSequence:
            for i in range(len(TO_timings)-1):
                # search for HS on same side as i
                foundHS = False
                side = events[TO_timings[i]][0]
                minimumChange = (0,1000)
                for j in range(TO_timings[i]+1, TO_timings[i+1]):
                    # print(f"DEBUG: {side.name} heel distance on frame {j} is {heel_distance[side][j]}")
                    if heel_distance[side][j] >= heel_distance[side][j+1] and heel_distance[side][j] > heel_distance[side][j-1]:
                        events[j-1] = (side, "HS")
                        foundHS = True
                        break
                    if heel_distance[side][j] > heel_distance[side][j-1] and minimumChange[1] > heel_distance[side][j] - heel_distance[side][j-1]:
                        # print(f"DEBUG: {side.name} new min heel distance on frame {j} is {heel_distance[side][j]}")
                        minimumChange = (j,  heel_distance[side][j] - heel_distance[side][j-1])
                if not foundHS:
                    # search minimum change less than 0.002
                    if minimumChange[1]< 200: #0.002:
                        print(f"WARN: Non peak HS event at {minimumChange[0]} frame with min change {minimumChange[1]}")
                        events[minimumChange[0]] = (side, "HS")
                        foundHS = True
                    else:
                        print(f"WARN: Failed to detect HS events between {TO_timings[i]} and {TO_timings[i+1]} frame")
                        print(TO[Side.LEFT], TO[Side.RIGHT])
                        break
                # search from last TO to end of video for optional HS
                if i==len(TO_timings)-2:
                    side = events[TO_timings[-1]][0]
                    for j in range(TO_timings[-1]+1, len(heel_distance[side])-1):
                        if heel_distance[side][j] >= heel_distance[side][j+1] and heel_distance[side][j] > heel_distance[side][j-1]:
                            events[j-1] = (side, "HS")
                            break 
            if foundHS:
                for side in Side: 
                    HS[side] = np.array([frame for frame,event in events.items() if event[0] == side and event[1] == "HS"])
                print_analysis_HSTO(HS, TO, use_gaitdata=False, fps=gaitdata.fps, kinedata=kinedata)
        else:
            print("WARN: Detected TO events not alternating")
            print(TO[Side.LEFT], TO[Side.RIGHT])

    else:
        print("WARN: Insufficient TO events to detect HS events.")


def cbta(gaitdata):

    #JSLOW(20230807) - manually selected HS TO frames
    if hasattr(gaitdata,"manualHS") and hasattr(gaitdata,"manualTO"):
        print("CBTA bypassed using manually selected HS TO events")
        print_analysis_HSTO(gaitdata.manualHS, gaitdata.manualTO, True, gaitdata)
        return

    ### !!! Assuming forwards is positive x
    fig_dir = "./"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    padlength = 5 # used for butterworth filter

    if (gaitdata.framework == "mediapipe") or (gaitdata.framework == "mediapipeheavy") or (gaitdata.framework == "arkit"):
        data = gaitdata.data_world
        distance_units = "m"
    elif gaitdata.framework == "vicon":
        data = gaitdata.data_world
        distance_units = "mm"
    else:
        data = gaitdata.data
        distance_units = "pixels"

    gaitdata.handle_missing_walk_direction()

    #JSLOW(20230706) - remedy jump noise
    if hasattr(gaitdata,"adjustOpenpose"):
        if gaitdata.adjustOpenpose == 2:
            gaitdata.remedy_jump_noise()

    display_raw = True

    heel_raw = {Side.LEFT: [], Side.RIGHT: []}
    heel_raw[Side.LEFT] = np.array(data[Joint.LEFT_HEEL] - data[Joint.MIDDLE_HIP])
    heel_raw[Side.RIGHT] = np.array(data[Joint.RIGHT_HEEL] - data[Joint.MIDDLE_HIP])
    toe_raw = {Side.LEFT: [], Side.RIGHT: []}
    toe_raw[Side.LEFT] = np.array(data[Joint.LEFT_TOE_BIG] - data[Joint.MIDDLE_HIP])
    toe_raw[Side.RIGHT] = np.array(data[Joint.RIGHT_TOE_BIG] - data[Joint.MIDDLE_HIP])
    #JSLOW(20230630) - adjust keypoints detected by Openpose
    if hasattr(gaitdata,"adjustOpenpose"):
        if gaitdata.adjustOpenpose == 1:
            adjustedKeypoints = openposeAdjustment([heel_raw, toe_raw], gaitdata.fps)
            heel_raw, toe_raw = adjustedKeypoints['heel'], adjustedKeypoints['toe']

    b, a = scipy.signal.butter(3, 7.0, 'lp', fs=gaitdata.fps, output='ba')

    heel_filt = {Side.LEFT: [], Side.RIGHT: []}
    toe_filt = {Side.LEFT: [], Side.RIGHT: []}
    
    #JSLOW(20230630) - added detrend step to handle small stride walking style
    for side in Side:
        heel_filt[side] = scipy.signal.filtfilt(b, a, scipy.signal.detrend(heel_raw[side],axis=0), axis=0, padlen=padlength)
        toe_filt[side] = scipy.signal.filtfilt(b, a, scipy.signal.detrend(toe_raw[side],axis=0), axis=0, padlen=padlength)

    stride_freq = {Side.LEFT: [], Side.RIGHT: []}
    f = {Side.LEFT: [], Side.RIGHT: []}
    psd = {Side.LEFT: [], Side.RIGHT: []}

    HS = {Side.LEFT: [], Side.RIGHT: []}
    TO = {Side.LEFT: [], Side.RIGHT: []}

    min_separation_coeff = 0.75  # Used to decide minimum duration between same event over different cycles

    # Get average leg length in windows to set min. peak height
    leglen_window_halfwidth = 15  # scale computation window halfwidth in frames
    leg_lengths = {Side.LEFT: [], Side.RIGHT: []}
    avg_leglen = {Side.LEFT: [], Side.RIGHT: []}

    for i in range(len(data[Joint.LEFT_HIP])):
        leg_lengths[Side.LEFT].append(
            np.linalg.norm(data[Joint.LEFT_HIP][i][:] - data[Joint.LEFT_KNEE][i][:]) + np.linalg.norm(
                data[Joint.LEFT_KNEE][i][:] - data[Joint.LEFT_ANKLE][i][:]))
        leg_lengths[Side.RIGHT].append(
            np.linalg.norm(data[Joint.RIGHT_HIP][i][:] - data[Joint.RIGHT_KNEE][i][:]) + np.linalg.norm(
                data[Joint.RIGHT_KNEE][i][:] - data[Joint.RIGHT_ANKLE][i][:]))
    for side in Side:
        for i in range(len(data[Joint.LEFT_HIP])):
            if i + 1 <= leglen_window_halfwidth:
                avg = np.average(leg_lengths[side][0:(2 * leglen_window_halfwidth)])
            elif len(data[Joint.LEFT_HIP]) - 1 <= i + leglen_window_halfwidth:
                avg = np.average(leg_lengths[side][len(data[Joint.LEFT_HIP]) - 1 - (2 * leglen_window_halfwidth):])
            else:
                avg = np.average(leg_lengths[side][i - leglen_window_halfwidth:i + leglen_window_halfwidth + 1])
            avg_leglen[side].append(avg)
        avg_leglen[side] = np.array(avg_leglen[side])

    for side in Side:
        f[side], psd[side] = scipy.signal.periodogram(heel_filt[side][:, 0], fs=gaitdata.fps)
        stride_freq[side] = f[side][np.argmax(psd[side])]
        print(side.name+" detected stride frequency: "+str(stride_freq[side])+"Hz")

    #JSLOW(20230714) - pick higher detected stride frequency if one side is too low
    if stride_freq[Side.LEFT]<0.1:
        stride_freq[Side.LEFT] = max(stride_freq[Side.LEFT], stride_freq[Side.RIGHT])
    if stride_freq[Side.RIGHT]<0.1:
        stride_freq[Side.RIGHT] = max(stride_freq[Side.LEFT], stride_freq[Side.RIGHT])

    for side in Side: 
    #JSLOW(20230630) - removed minimum distance for peak to handle small step walking 
        if gaitdata.walk_direction == Side.RIGHT:
            #HS[side], _ = scipy.signal.find_peaks(heel_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff)* gaitdata.fps, height=avg_leglen[side]*0.20, prominence=avg_leglen[side]*0.1)
            #TO[side], _ = scipy.signal.find_peaks(-toe_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff) * gaitdata.fps, height=avg_leglen[side]*0.15, prominence=avg_leglen[side]*0.1)
            HS[side], _ = scipy.signal.find_peaks(heel_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff)* gaitdata.fps, prominence=avg_leglen[side]*0.1)
            TO[side], _ = scipy.signal.find_peaks(-toe_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff) * gaitdata.fps, prominence=avg_leglen[side]*0.1)
        elif gaitdata.walk_direction == Side.LEFT:
            #HS[side], _ = scipy.signal.find_peaks(-heel_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff)* gaitdata.fps, height=avg_leglen[side]*0.20, prominence=avg_leglen[side]*0.1)
            #TO[side], _ = scipy.signal.find_peaks(toe_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff) * gaitdata.fps, height=avg_leglen[side]*0.15, prominence=avg_leglen[side]*0.1)
            HS[side], _ = scipy.signal.find_peaks(-heel_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff)* gaitdata.fps, prominence=avg_leglen[side]*0.1)
            TO[side], _ = scipy.signal.find_peaks(toe_filt[side][:, 0], distance=((1/stride_freq[side]) * min_separation_coeff) * gaitdata.fps, prominence=avg_leglen[side]*0.1)

    time_axis = np.linspace(0, len(heel_filt[side][:, 0]) / gaitdata.fps, num=len(heel_filt[side][:, 0]))

    if display_raw:

        temp_title = gaitdata.framework+" "+"Raw x-distance from sacrum"
        fig_raw, axs = plt.subplots(2, num=temp_title)
        for ax_idx, side in enumerate(Side):
            axs[ax_idx].plot(time_axis, heel_raw[side][:, 0], label='Heel')
            axs[ax_idx].plot(time_axis, toe_raw[side][:, 0], label='Toe')
            axs[ax_idx].set(ylabel=side.name + " x-distance (" + distance_units + ")")
            axs[ax_idx].set(xlabel="Time (seconds)")
            axs[ax_idx].label_outer()
            axs[ax_idx].legend()

        plt.gcf().savefig(fig_dir + temp_title)

    temp_title = gaitdata.framework+" "+"Filtered x-distance from sacrum"
    fig_filt, axs = plt.subplots(2, num=temp_title)
    for ax_idx, side in enumerate(Side):

        axs[ax_idx].plot(time_axis, heel_filt[side][:, 0], label='Heel')
        axs[ax_idx].plot(time_axis, toe_filt[side][:, 0], label='Toe')
        axs[ax_idx].plot(time_axis[HS[side]], heel_filt[side][HS[side], 0], "x")
        for i, x in enumerate(HS[side]):
            axs[ax_idx].annotate("HS", (time_axis[x], heel_filt[side][x, 0]))
        axs[ax_idx].plot(time_axis[TO[side]], toe_filt[side][TO[side], 0], "x")
        for i, x in enumerate(TO[side]):
            axs[ax_idx].annotate("TO", (time_axis[x], toe_filt[side][x, 0]))

        axs[ax_idx].set(ylabel=side.name+" x-distance ("+distance_units+")")

        axs[ax_idx].set(xlabel="Time (seconds)")
        axs[ax_idx].label_outer()
        axs[ax_idx].legend()

        plt.gcf().savefig(fig_dir + temp_title)

    csvfile = open('cbta_dist.csv', 'w', newline='')
    cbtwriter = csv.writer(csvfile, lineterminator='\n')
    cbtwriter.writerow(["time", "left_heel", "left_toe", "right_heel", "right_toe"])
    xDistances = np.array([time_axis, heel_filt[Side.LEFT][:, 0], toe_filt[Side.LEFT][:, 0], heel_filt[Side.RIGHT][:, 0], toe_filt[Side.RIGHT][:, 0]]).T
    # xDistances = np.array([time_axis, heel_raw[Side.LEFT][:, 0], toe_raw[Side.LEFT][:, 0], heel_raw[Side.RIGHT][:, 0], toe_raw[Side.RIGHT][:, 0]]).T
    cbtwriter.writerows(xDistances)

    print_analysis_HSTO(HS, TO, True, gaitdata)

# old
def foot_velocity_algorithm(gaitdata):

    data = gaitdata.data

    y_left_midpoint = data[Joint.LEFT_HEEL][:, 1] + 0.5*(data[Joint.LEFT_TOES][:, 1] - data[Joint.LEFT_HEEL][:, 1])
    y_left_midpoint = y_left_midpoint - data[Joint.MIDDLE_HIP][:, 1]
    y_right_midpoint = data[Joint.RIGHT_HEEL][:, 1] + 0.5*(data[Joint.RIGHT_TOES][:, 1] - data[Joint.RIGHT_HEEL][:, 1])
    y_right_midpoint = y_right_midpoint - data[Joint.MIDDLE_HIP][:, 1]

    b, a = scipy.signal.butter(2, 3.0, 'lp', fs=gaitdata.fps, output='ba')

    y_left_midpoint_smoothed = scipy.signal.filtfilt(b, a, y_left_midpoint, padlen=10)
    y_right_midpoint_smoothed = scipy.signal.filtfilt(b, a, y_right_midpoint, padlen=10)

    time_delta = 1/gaitdata.fps

    vel_left = np.gradient(y_left_midpoint_smoothed, time_delta)
    vel_right = np.gradient(y_right_midpoint_smoothed, time_delta)

    welch_f_left, welch_psd_left = scipy.signal.welch(vel_left, fs=gaitdata.fps)
    stride_period_l = 1 / welch_f_left[np.argmax(welch_psd_left)] # in seconds
    print("Stride period left: "+str(stride_period_l)+" seconds")

    toeoff_left, _ = scipy.signal.find_peaks(vel_left, distance=gaitdata.fps*(0.85*stride_period_l))

    heelstrike_left_candidates, _ = scipy.signal.find_peaks(-vel_left, distance=gaitdata.fps*(0.85*stride_period_l))
    heelstrike_left = []
    y_left_min = np.amin(y_left_midpoint_smoothed)
    y_left_max = np.amax(y_left_midpoint_smoothed)
    for idx in heelstrike_left_candidates:
        if y_left_midpoint_smoothed[idx] < 0.35*(y_left_max-y_left_min) + y_left_min:
            heelstrike_left.append(idx)

    toeoff_right, _ = scipy.signal.find_peaks(vel_right, distance=gaitdata.fps * (0.85 * stride_period_l))

    heelstrike_right_candidates, _ = scipy.signal.find_peaks(-vel_right, distance=gaitdata.fps*(0.85*stride_period_l))
    heelstrike_right = []
    y_right_min = np.amin(y_right_midpoint_smoothed)
    y_right_max = np.amax(y_right_midpoint_smoothed)
    for idx in heelstrike_right_candidates:
        if y_right_midpoint_smoothed[idx] < 0.35*(y_right_max-y_right_min) + y_right_min:
            heelstrike_right.append(idx)

    time_axis = np.linspace(0, len(vel_left) / gaitdata.fps, num=len(vel_left))

    fig_l = plt.figure("Left foot fva")
    plt.plot(time_axis, y_left_midpoint)
    plt.plot(time_axis, y_left_midpoint_smoothed)
    plt.plot(time_axis, vel_left)
    plt.plot(time_axis[heelstrike_left], vel_left[heelstrike_left], "xm")
    plt.plot(time_axis[toeoff_left], vel_left[toeoff_left], "xr")
    plt.xlabel("Time (seconds)")

    fig_r = plt.figure("Right foot fva")
    plt.plot(time_axis, y_right_midpoint)
    plt.plot(time_axis, y_right_midpoint_smoothed)
    plt.plot(time_axis, vel_right)
    plt.plot(time_axis[heelstrike_right], vel_right[heelstrike_right], "xm")
    plt.plot(time_axis[toeoff_right], vel_right[toeoff_right], "xr")
    plt.xlabel("Time (seconds)")

    # fig_welch = plt.figure("Welch left")
    # plt.plot(welch_f_left, welch_psd_left)

def shank_ang_vel_analysis(gaitdata):

    #JSLOW(20230807) - manually selected HS TO frames
    if hasattr(gaitdata,"manualHS") and hasattr(gaitdata,"manualTO"):
        print("SAV bypassed using manually selected HS TO events")
        print_analysis_HSTO(gaitdata.manualHS, gaitdata.manualTO, True, gaitdata)
        return
    
    # fig_dir = "A:/Users/Kin/Desktop/plots/"

    # if not os.path.exists(fig_dir):
    #     os.makedirs(fig_dir)

    padlength = 3 # padlength used for butterworth filter

    if (gaitdata.framework == "mediapipe") or (gaitdata.framework == "mediapipeheavy") or (gaitdata.framework == "vicon") or (gaitdata.framework == "arkit"):
        data = gaitdata.data_world
    else:
        data = gaitdata.data

    gaitdata.handle_missing_walk_direction()

    # Find medio-lateral angle of shank with respect to the vertical. Ankle to the right of the knee will give a positive angle.
    def find_ml_angle(knee, ankle):
        x = ankle[0] - knee[0]
        y = ankle[1] - knee[1]
        angle = np.arcsin(x / np.sqrt(x**2 + y**2))
        return angle

    angle_raw = {Side.LEFT: [], Side.RIGHT: []}

    # b, a = scipy.signal.butter(2, 10.0, 'lp', fs=gaitdata.fps, output='ba')
    b, a = scipy.signal.butter(3, 7.0, 'lp', fs=gaitdata.fps, output='ba')

    filt_knee = {Side.LEFT: [], Side.RIGHT: []}
    filt_ank = {Side.LEFT: [], Side.RIGHT: []}
    filt_knee[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_KNEE], axis=0, padlen=padlength)
    filt_ank[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_ANKLE], axis=0, padlen=padlength)
    filt_knee[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_KNEE], axis=0, padlen=padlength)
    filt_ank[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_ANKLE], axis=0, padlen=padlength)

    for f in range(np.shape(data[Joint.LEFT_KNEE])[0]):
        if gaitdata.walk_direction == Side.RIGHT:
            angle_raw[Side.LEFT].append(find_ml_angle(data[Joint.LEFT_KNEE][f], data[Joint.LEFT_ANKLE][f]))
            angle_raw[Side.RIGHT].append(find_ml_angle(data[Joint.RIGHT_KNEE][f], data[Joint.RIGHT_ANKLE][f]))
        elif gaitdata.walk_direction == Side.LEFT:
            angle_raw[Side.LEFT].append(-find_ml_angle(data[Joint.LEFT_KNEE][f], data[Joint.LEFT_ANKLE][f]))
            angle_raw[Side.RIGHT].append(-find_ml_angle(data[Joint.RIGHT_KNEE][f], data[Joint.RIGHT_ANKLE][f]))

    for side in Side:
        angle_raw[side] = scipy.signal.filtfilt(b, a, angle_raw[side], padlen=padlength)

    avel_raw = {Side.LEFT: [], Side.RIGHT: []}
    for side in Side:
        avel_raw[side] = np.gradient(angle_raw[side]) * gaitdata.fps
    avel_use = avel_raw # may not need filter for avel_raw since coordinates were already filtered

    ###############################
    ### Gait phase events ###
    ###############################

    stride_freq = {Side.LEFT: [], Side.RIGHT: []}
    f = {Side.LEFT: [], Side.RIGHT: []}
    psd = {Side.LEFT: [], Side.RIGHT: []}

    freq_tester = {Side.LEFT: [], Side.RIGHT: []}
    freq_tester[Side.LEFT] = np.array(data[Joint.LEFT_ANKLE] - data[Joint.MIDDLE_HIP])
    freq_tester[Side.LEFT] = scipy.signal.filtfilt(b, a, freq_tester[Side.LEFT], axis=0, padlen=padlength)
    freq_tester[Side.RIGHT] = np.array(data[Joint.RIGHT_ANKLE] - data[Joint.MIDDLE_HIP])
    freq_tester[Side.RIGHT] = scipy.signal.filtfilt(b, a, freq_tester[Side.RIGHT], axis=0, padlen=padlength)

    for side in Side:

        f[side], psd[side] = scipy.signal.periodogram(freq_tester[side][:, 0], fs=gaitdata.fps) # Use heel x-coord to detect frequency
        stride_freq[side] = f[side][np.argmax(psd[side])]

    HSTO = shank_ang_vel_ED(avel_use, stride_freq, gaitdata.fps, gaitdata.framework)
    print_analysis_HSTO(HSTO[GaitEvent.HEELSTRIKE], HSTO[GaitEvent.TOEOFF], True, gaitdata)

def shank_ang_vel_imu(csv_left: str, csv_right: str, video_start=0, video_end=99999, fps=50):

    csv_files = {}
    csv_files[Side.LEFT] = csv_left
    csv_files[Side.RIGHT] = csv_right

    b, a = scipy.signal.butter(3, 7.0, 'lp', fs=fps, output='ba')
    padlength = 3

    avel_raw = {Side.LEFT: [], Side.RIGHT: []}
    avel_use = {Side.LEFT: [], Side.RIGHT: []}

    for side in Side:
        start = False
        row_num = 0
        with open(csv_files[side], newline='', errors='ignore') as csvfile:
            csvreader = csv.reader(csvfile)
            for row in csvreader:

                row_num += 1

                if row_num == 1:
                    continue

                if not start and float(row[0]) >= video_start:
                    start = True
                    start_count = 1
                    print(side.name + " started at " + str(row[0]))
                if float(row[0]) > video_end:
                    print(side.name + " ended before " + str(row[0]))
                    break
                if start:
                    start_count += 1
                    # if (start_count+2) % 4 == 1: # To reduce sample frequency
                    # if start_count % 2 == 1: # To handle duplicate rows or reduce frequency
                    if True:
                        avel_raw[side].append(np.deg2rad(float(row[3]) / 14.375)) # constant of 14.375 according to gyro manufacturer's specifications: https://wiki.seeedstudio.com/Grove-3-Axis_Digital_Gyro/
        avel_raw[side] = np.array(avel_raw[side])

    avel_raw[Side.RIGHT] = -avel_raw[Side.RIGHT] # as of now, with regards to sign, the left sensor has the "correct" orientation while the right sensor has "opposite" when worn on the foot
    # If all rows are used, total number of rows may be uneven (because devices stopped recording at slightly different times?). Trim off 'excess' rows.
    row_limit = min([len(avel_raw[Side.RIGHT]), len(avel_raw[Side.LEFT])])
    for side in Side:
        avel_raw[side] = avel_raw[side][0:row_limit]

    est_freq = {Side.LEFT: [], Side.RIGHT: []}
    f = {Side.LEFT: [], Side.RIGHT: []}
    psd = {Side.LEFT: [], Side.RIGHT: []}

    for side in Side:

        avel_use[side] = scipy.signal.filtfilt(b, a, avel_raw[side], padlen=padlength)

        f[side], psd[side] = scipy.signal.periodogram(avel_use[side][:], fs=fps) # Use gyroscope data to estimate frequency
        psd_max = np.max(psd[side])

        freq_peaks, _ = scipy.signal.find_peaks(psd[side], height=psd_max*0.75)
        est_freq[side] = f[side][freq_peaks[0]] # angular velocity seems to usually have 2 peaks. use the lower-frequency peak as it's closer to stride frequency.

        plt.plot(f[side], psd[side])
        print("Est. "+side.name+" frequency: "+str(est_freq[side])+"hz")

    HSTO = shank_ang_vel_ED(avel_use, est_freq, fps, title_label="IMU")
    print_analysis_HSTO(HSTO[GaitEvent.HEELSTRIKE], HSTO[GaitEvent.TOEOFF], False)

# Event detection using the shank angular velocity (sagittal). Returns heelstrikes and toe-offs.
def shank_ang_vel_ED(avel_use: dict, est_freq: dict, fps, title_label="Gaitanalysis") -> dict:
    """

    :rtype: dict with GaitEvent keys (heelstrike, toeoff) and array values containing frame number of events
    :param avel_use: Angular velocity dict with gaitevents.Side keys
    :param est_freq: Estimated frequency of stride used as a gauge for min/max spacing between events
    :param title_label:
    """

    def find_zero_crossings(array):
        # Finds crossings from negative to positive
        sign_diff = np.diff(np.sign(array))
        for i in range(len(sign_diff)):
            if sign_diff[i] < 0:
                sign_diff[i] = 0
# original code: get index of non zero element in array. Shape (1,n) convert to shape(n,) by [0]
        candidates = np.where(sign_diff)[0]
# JSLOW (09/11/2022): alternative method
        # candidates = np.nonzero(sign_diff)[0]
        final_results = []
# idx is index of array where element is negative/zero and next element is zero/positive
# final_results is index of current or next element whichever is closer to zero 
        for idx in candidates:
            if abs(array[idx]) < abs(array[idx+1]):
                final_results.append(idx)
            else:
                final_results.append(idx+1)
        return final_results

    fig_dir = "./"

    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    HS = {Side.LEFT: [], Side.RIGHT: []}
    TO = {Side.LEFT: [], Side.RIGHT: []}

    Msw_onset = {Side.LEFT: [], Side.RIGHT: []}
    Tsw_onset = {Side.LEFT: [], Side.RIGHT: []}

    # tsw_min_height = 3  # for HomeRehab IMU
    # to_min_height = 1   # for HomeRehab IMU
    tsw_min_height = 0.1  # for Metawear
    to_min_height = 0.1   # for Metawear
    tsw_width = 4
    to_width = 2
    min_sep_coeff = 0.75 # Used to decide minimum duration between same event over different cycles
    max_sep_coeff = 1.25

    est_period = {Side.LEFT: [], Side.RIGHT: []}

    # Add events for frames between Tsw events (terminal swing onsets)
    for side in Side:

        est_period[side] = 1 / est_freq[side]
        # plt.clf()
        # plt.plot(avel_use[side])
        # plt.savefig("DEBUG_Plot_"+side.name+".png")
        Tsw_onset[side], _ = scipy.signal.find_peaks(avel_use[side], height=tsw_min_height, distance=est_period[side]*fps*min_sep_coeff, width=tsw_width)
        # print("DEBUG: len(Tsw_onset[side]) = " + str(len(Tsw_onset[side])))
        for i in range(len(Tsw_onset[side])-1):
            start = Tsw_onset[side][i]
            end = Tsw_onset[side][i + 1]
            HS[side].append(start + find_zero_crossings(-avel_use[side][start:end])[0])
            TO_candidates, _ = scipy.signal.find_peaks(-avel_use[side][start:end], height=-to_min_height, width=to_width)
            TO[side].append(start + TO_candidates[-1])
            # TO[side].append(start + find_zero_crossings(avel_use[side][start:end])[0]) # tweak
            Msw_onset[side].append(start + find_zero_crossings(avel_use[side][start:end])[-1])

    # Add TO before first Tsw, both sides
    for side in Side:
        
        TO_candidates, _ = scipy.signal.find_peaks(-avel_use[side][0:Tsw_onset[side][0]], height=-to_min_height, width=to_width)
        if len(TO_candidates) > 0:
            TO[side].insert(0, TO_candidates[-1])
        # TO_candidates = find_zero_crossings(avel_use[side][0:Tsw_onset[side][0]])
        # if len(TO_candidates) > 0:
        #     TO[side].insert(0, TO_candidates[-1])  # tweak

    # Add HS after last Tsw, both sides
    for side in Side:
        HS_candidates = find_zero_crossings(-avel_use[side][Tsw_onset[side][-1]:])
        if len(HS_candidates) > 0:
            HS[side].append(Tsw_onset[side][-1] + HS_candidates[0])

    # Add TO after last Tsw, both sides
    for side in Side:
        TO_candidates, _ = scipy.signal.find_peaks(-avel_use[side][Tsw_onset[side][-1]:], height=-to_min_height, width=to_width)
        if len(TO_candidates) > 0:
            # if (len(TO[side]) >= 2 and ((TO_candidates[-1]+Tsw_onset[side][-1]) - TO[side][-1]) >= min_sep_coeff*np.mean(np.diff(TO[side])))\
            #         or (len(HS[side]) >= 2 and ((TO_candidates[-1]+Tsw_onset[side][-1]) - TO[side][-1]) >= min_sep_coeff*np.mean(np.diff(HS[side]))):
            if ((TO_candidates[-1] + Tsw_onset[side][-1]) - TO[side][-1] >= est_period[side]*fps*min_sep_coeff) and \
                    ((TO_candidates[-1] + Tsw_onset[side][-1]) - TO[side][-1] <= est_period[side]*fps*max_sep_coeff):
                TO[side].append(TO_candidates[-1]+Tsw_onset[side][-1])
        # TO_candidates = find_zero_crossings(avel_use[side][Tsw_onset[side][-1]:]) # tweak
        # if len(TO_candidates) > 0:
        #     if len(TO[side]) >= 2 and ((TO_candidates[-1]+Tsw_onset[side][-1]) - TO[side][-1]) > min_sep_coeff*(TO[side][-1] - TO[side][-2]):
        #         TO[side].append(TO_candidates[-1]+Tsw_onset[side][-1])

    # Add HS before first Tsw, both sides
    # Can cause problems if there is no corresponding opposite toeoff
    # Chance of false positive is high when signal is noisy
    # for side in Side:
    #     HS_candidates = find_zero_crossings(-avel_use[side][0:Tsw_onset[side][0]])
    #     if len(HS_candidates) > 0:
    #         # if len(HS[side]) >= 2 and (HS[side][0] - HS_candidates[0]) > min_sep_coeff*(HS[side][1] - HS[side][0]): # Check if candidate is too close to next HS
    #         if stride_period[side]*fps*min_sep_coeff <= (HS[side][0] - HS_candidates[0]) <= \
    #                 stride_period[side]*fps*max_sep_coeff:
    #             HS[side].insert(0, HS_candidates[0])

    # Add Msw onset before first Tsw, both sides
    for side in Side:
        Msw_onset_candidates = find_zero_crossings(avel_use[side][0:Tsw_onset[side][0]])
        if len(Msw_onset_candidates) > 0:
            Msw_onset[side].insert(0, Msw_onset_candidates[-1])

    time_axis = np.linspace(0, len(avel_use[Side.LEFT]) / fps, num=len(avel_use[Side.LEFT])) # Use time axis from left side for both sides

    ###############################
    ### Plots ###
    ###############################

    # temp_title = title_label+" "+"Raw shank angle"
    # fig_angle, axs = plt.subplots(2, num=temp_title)
    # plt.gcf().set(label=title_label+" "+"Raw shank angle")
    # axs[0].plot(time_axis, avel_raw[Side.LEFT])
    # axs[0].set(ylabel="Left foot angle (rad)")
    # axs[1].plot(time_axis, avel_raw[Side.RIGHT])
    # axs[1].set(ylabel="Right foot angle (rad)")
    # for ax in axs:
    #     ax.set(xlabel="Time (seconds)")
    #     ax.label_outer()
    # plt.gcf().savefig(fig_dir+temp_title)

    temp_title = title_label+" "+"Shank angular velocity"
    fig_angv, axs = plt.subplots(2, num=temp_title, figsize=(15, 9))
    axs[0].set(ylabel="Left shank ang. vel. (rad/s)")
    axs[1].set(ylabel="Right shank ang. vel. (rad/s)")
    axs[0].set_title(f"Shank Angular Velocity - stride period({est_period[Side.LEFT]:.3f}s)")
    axs[1].set_title(f"Shank Angular Velocity - stride period({est_period[Side.RIGHT]:.3f}s)")
    for ax in axs:
        ax.set(xlabel="Time (seconds)")
        ax.label_outer()

    for ax_idx, side in enumerate(Side):
        axs[ax_idx].plot(time_axis, avel_use[side])
        axs[ax_idx].plot(time_axis[HS[side]], avel_use[side][HS[side]], "x")
        for i, x in enumerate(HS[side]):
            axs[ax_idx].annotate("HS", (time_axis[x], avel_use[side][x]))
        axs[ax_idx].plot(time_axis[Tsw_onset[side]], avel_use[side][Tsw_onset[side]], "x")
        for i, x in enumerate(Tsw_onset[side]):
            axs[ax_idx].annotate("Tsw onset", (time_axis[x], avel_use[side][x]))
        axs[ax_idx].plot(time_axis[TO[side]], avel_use[side][TO[side]], "x")
        for i, x in enumerate(TO[side]):
            axs[ax_idx].annotate("TO", (time_axis[x], avel_use[side][x]))
        axs[ax_idx].plot(time_axis[Msw_onset[side]], avel_use[side][Msw_onset[side]], "x")
        for i, x in enumerate(Msw_onset[side]):
            axs[ax_idx].annotate("Msw onset", (time_axis[x], avel_use[side][x]))
        # Mst onset = contralateral TO
        axs[ax_idx].plot(time_axis[TO[side.opposite()]], avel_use[side][TO[side.opposite()]], "x")
        for i, x in enumerate(TO[side.opposite()]):
            axs[ax_idx].annotate("Mst onset", (time_axis[x], avel_use[side][x]))
        # Tst onset = contralateral Tsw
        axs[ax_idx].plot(time_axis[Tsw_onset[side.opposite()]], avel_use[side][Tsw_onset[side.opposite()]], "x")
        for i, x in enumerate(Tsw_onset[side.opposite()]):
            axs[ax_idx].annotate("Tst onset", (time_axis[x], avel_use[side][x]))
        # Psw onset = contralateral HS
        axs[ax_idx].plot(time_axis[HS[side.opposite()]], avel_use[side][HS[side.opposite()]], "x")
        for i, x in enumerate(HS[side.opposite()]):
            axs[ax_idx].annotate("Psw onset", (time_axis[x], avel_use[side][x]))

    plt.gcf().savefig(fig_dir + temp_title)

    HSTO = {}
    HSTO[GaitEvent.HEELSTRIKE] = HS
    HSTO[GaitEvent.TOEOFF] = TO

    # Added the following to verify HSTO detection visually
    print("Heelstrike:", HS)
    print("Toe Off:", TO)

    plt.close()
    leftEvents = np.array([HS[Side.LEFT], [1]*len(HS[Side.LEFT])]).transpose()
    plt.scatter(leftEvents[:,0], leftEvents[:,1], marker="o", label="Left Heel Strike")
    y=0
    for x in leftEvents[:,0]:
        plt.text(x=x, y = 0.5 + (-0.1)*y, s=f"{x}\n{x/fps:.3f}s")
        y += 1
        if y > 1:
          y = 0
    HSevents = np.copy(leftEvents)
    leftEvents = np.array([TO[Side.LEFT], [1]*len(TO[Side.LEFT])]).transpose()
    plt.scatter(leftEvents[:,0], leftEvents[:,1], marker=">", label="Left Toe Off")
    rightEvents = np.array([HS[Side.RIGHT], [-1]*len(HS[Side.RIGHT])]).transpose()
    plt.scatter(rightEvents[:,0], rightEvents[:,1], marker="o", label="Right Heel Strike")
    y=0
    for x in rightEvents[:,0]:
        plt.text(x=x, y = -1.5 + (-0.1)*y, s=f"{x}\n{x/fps:.3f}s")
        y += 1
        if y > 1:
          y = 0
    HSevents = np.append(HSevents, rightEvents, axis=0)
    rightEvents = np.array([TO[Side.RIGHT], [-1]*len(TO[Side.RIGHT])]).transpose()
    plt.scatter(rightEvents[:,0], rightEvents[:,1], marker=">", label="Right Toe Off")

    HSevents = HSevents[HSevents[:,0].argsort()]
    plt.plot(HSevents[:,0], HSevents[:,1], alpha=0.5)

    plt.xlabel('Frame')
    plt.ylim([-4,4])
    plt.yticks([-1,0,1])
    
    plt.title('Gait Events Detected')
    plt.legend()
    # plt.show()
    figure = plt.gcf()
    figure.set_size_inches(19.20, 10.80)
    plt.savefig("checkkGait.png", dpi=100)
    return HSTO

# Prints several gait parameters, given the heelstrike and toeoff frames.
# Assumes subject is walking in positive x direction.
# def print_analysis_HSTO(HS, TO, use_gaitdata: bool, gaitdata = GaitData("none", 0), fps=50):
def print_analysis_HSTO(HS, TO, use_gaitdata: bool, gaitdata = None, fps=50, kinedata={}):

    def find_pairs(first, second):
        pairs = []
        first_track = 0
        second_track = 0
        while (second_track < len(second)) & (first_track < len(first)):
            if second[second_track] >= first[first_track]:
                # take the latest possible of the 'first' events
                for i in range(first_track, len(first)):
                    if first[i] > second[second_track]:
                        first_track = i-1
                        break
                    elif i == len(first)-1:
                        first_track = i
                if second[second_track] == first[first_track]:
                    print("WARNING: Two events found on same frame while finding pairs.")
                pairs.append([first[first_track], second[second_track]])
                first_track += 1
            second_track += 1
        return pairs

    # Same as find_pairs, but only allow for 2nd event to occur after 1st
    def find_pairs_strict(first, second):
        pairs = []
        first_track = 0
        second_track = 0
        while (second_track < len(second)) & (first_track < len(first)):
            if second[second_track] > first[first_track]:
                # take the latest possible of the 'first' events
                for i in range(first_track, len(first)-1):
                    if first[i+1] >= second[second_track]:
                        first_track = i
                        break
                pairs.append([first[first_track], second[second_track]])
                first_track += 1
            second_track += 1
        return pairs

    # Tests if event falls within any of the pairs. Returns boolean.
    def event_in_pairs(event, pairs):
        for i in range(len(pairs)):
            if event >= pairs[i][0] and event <= pairs[i][1]:
                return True
        return False

    def is_pair_inside_pair(pair_inside, pair_outside):
        return (pair_inside[0] >= pair_outside[0] and pair_inside[0] <= pair_outside[1]) and (pair_inside[1] >= pair_outside[0] and pair_inside[1] <= pair_outside[1])

    fig_dir = "./"
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    if use_gaitdata:
        fps = gaitdata.fps
    b, a = scipy.signal.butter(6, 3.0, 'lp', fs=fps, output='ba')
    padlength = 3

    if use_gaitdata:

        if (gaitdata.framework == "mediapipe") or (gaitdata.framework == "mediapipeheavy") or (gaitdata.framework == "arkit"):
            data = gaitdata.data_world
            distance_units = "m"
        elif gaitdata.framework == "vicon":
            data = gaitdata.data_world
            distance_units = "mm"
        else:
            data = gaitdata.data
            if "leg_length_left" in gaitdata.patientinfo.keys() and "leg_length_right" in gaitdata.patientinfo.keys():
                leg_lengths = {Side.LEFT: gaitdata.patientinfo["leg_length_left"], Side.RIGHT: gaitdata.patientinfo["leg_length_right"]}
                distance_units = "m"
            else:
                distance_units = " pixels"

        gaitdata.handle_missing_walk_direction()

        filt_hip = {Side.LEFT: [], Side.RIGHT: []}
        filt_knee = {Side.LEFT: [], Side.RIGHT: []}
        filt_ank = {Side.LEFT: [], Side.RIGHT: []}
        filt_heel = {Side.LEFT: [], Side.RIGHT: []}
        filt_toe = {Side.LEFT: [], Side.RIGHT: []}
        filt_midshoulder = {Side.LEFT: [], Side.RIGHT: []}

        filt_hip[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_HIP], axis=0, padlen=padlength)
        filt_hip[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_HIP], axis=0, padlen=padlength)
        filt_knee[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_KNEE], axis=0, padlen=padlength)
        filt_ank[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_ANKLE], axis=0, padlen=padlength)
        filt_knee[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_KNEE], axis=0, padlen=padlength)
        filt_ank[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_ANKLE], axis=0, padlen=padlength)

        filt_heel[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_HEEL], axis=0, padlen=padlength)
        filt_heel[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_HEEL], axis=0, padlen=padlength)
        filt_toe[Side.LEFT] = scipy.signal.filtfilt(b, a, data[Joint.LEFT_TOE_BIG], axis=0, padlen=padlength)
        filt_toe[Side.RIGHT] = scipy.signal.filtfilt(b, a, data[Joint.RIGHT_TOE_BIG], axis=0, padlen=padlength)
        if gaitdata.framework != 'vicon':
            # JSLOW(20230529) added midhip for hip flexion
            filt_midhip = scipy.signal.filtfilt(b, a, data[Joint.MIDDLE_HIP], axis=0, padlen=padlength)
            filt_midshoulder = scipy.signal.filtfilt(b, a, data[Joint.MID_SHOULDER], axis=0, padlen=padlength)

        # Get length conversion scales
        if not gaitdata.is_world_units():
            scale_window_halfwidth = 15 # scale computation window halfwidth in frames

            leg_lengths_pixel = {Side.LEFT: [], Side.RIGHT: []}
            metres_per_pixel = {Side.LEFT: [], Side.RIGHT: []}

            for side in Side:
                for i in range(len(data[Joint.LEFT_HIP])):
                    leg_lengths_pixel[side].append(np.linalg.norm(filt_hip[side][i][:] - filt_knee[side][i][:]) + np.linalg.norm(filt_knee[side][i][:] - filt_ank[side][i][:]))
                for i in range(len(data[Joint.LEFT_HIP])):
                    if i+1 <= scale_window_halfwidth:
                        scale = leg_lengths[side] / np.average(leg_lengths_pixel[side][0:(2*scale_window_halfwidth)])
                    elif len(data[Joint.LEFT_HIP])-1 <= i+scale_window_halfwidth:
                        scale = leg_lengths[side] / np.average(leg_lengths_pixel[side][len(data[Joint.LEFT_HIP]) - 1 - (2 * scale_window_halfwidth):])
                    else:
                        scale = leg_lengths[side] / np.average(leg_lengths_pixel[side][i-scale_window_halfwidth:i+scale_window_halfwidth+1])
                    metres_per_pixel[side].append(scale)


    tohs_pairs = {Side.LEFT: [], Side.RIGHT: []}
    tohs_pairs_alt = {Side.LEFT: [], Side.RIGHT: []}
    hsto_pairs_alt = {Side.LEFT: [], Side.RIGHT: []}
    step_pairs = {Side.LEFT: [], Side.RIGHT: []}
    stride_pairs = {Side.LEFT: [], Side.RIGHT: []}

    for side in Side:
        tohs_pairs[side] = find_pairs(TO[side], HS[side])
        tohs_pairs_alt[side] = find_pairs(TO[side], HS[side.opposite()])
        hsto_pairs_alt[side] = find_pairs(HS[side], TO[side.opposite()])
        step_pairs[side] = find_pairs(HS[side.opposite()], HS[side])
        # stride_pairs[side] = find_pairs_strict(HS[side], HS[side])  # see below change on 18012023
        print(step_pairs[side])

    # JSLOW 18012023: change stride_pairs to construct from first element of opposite step_pairs
    #                 add on second element of last same side step pair if different from last heel strike
    for side in Side:
        step_HS = [sp[0] for sp in step_pairs[side.opposite()]]
        if step_HS[-1] != step_pairs[side][-1][1]:
            step_HS.append(step_pairs[side][-1][1])
        stride_pairs[side] = find_pairs_strict(step_HS, step_HS)

    print("Left steps (alternate heelstrike pairs): "+str(len(step_pairs[Side.LEFT])))
    print("Right steps (alternate heelstrike pairs): " + str(len(step_pairs[Side.RIGHT])))
    print("Left strides (full gait cycles): " + str(len(stride_pairs[Side.LEFT])))
    print(stride_pairs[Side.LEFT])
    print("Right strides (full gait cycles): " + str(len(stride_pairs[Side.RIGHT])))
    print(stride_pairs[Side.RIGHT])

    print("Heelstrike frames, left and right")
    print(HS[Side.LEFT])
    print(HS[Side.RIGHT])
    print("Toeoff frames, left and right")
    print(TO[Side.LEFT])
    print(TO[Side.RIGHT])

    step_lengths = {Side.LEFT: [], Side.RIGHT: []}
    step_durations = {Side.LEFT: [], Side.RIGHT: []}
    stride_lengths = {Side.LEFT: [], Side.RIGHT: []}
    stride_durations = {Side.LEFT: [], Side.RIGHT: []}

    init_double_supp_durations = {Side.LEFT: [], Side.RIGHT: []}
    term_double_supp_durations = {Side.LEFT: [], Side.RIGHT: []}

    csvfile = open('gaitanalysis.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, lineterminator='\n')

    for side in Side:

        print("\n"+side.name+" SIDE PARAMETERS:")
        print("********************")

        csvwriter.writerow([side.name])

        print("\nStride lengths and/or durations: ")
        for i in range(len(step_pairs[side.opposite()])):
            # Start with step of opposite foot
            if use_gaitdata:
                if gaitdata.is_world_units():
                    stride_length = filt_heel[side.opposite()][step_pairs[side.opposite()][i][1]][0] - filt_heel[side][step_pairs[side.opposite()][i][1]][0]
                else:
                    stride_length = (filt_heel[side.opposite()][step_pairs[side.opposite()][i][1]][0] * metres_per_pixel[side.opposite()][step_pairs[side.opposite()][i][1]]) - \
                                    (filt_heel[side][step_pairs[side.opposite()][i][1]][0] * metres_per_pixel[side.opposite()][step_pairs[side.opposite()][i][1]])
            stride_duration = (step_pairs[side.opposite()][i][1] - step_pairs[side.opposite()][i][0]) / fps
            for k in range(len(step_pairs[side])):
                # Add step of main foot
                if step_pairs[side][k][1] > step_pairs[side.opposite()][i][1]:
                    if use_gaitdata:
                        if gaitdata.is_world_units():
                            stride_length += filt_heel[side][step_pairs[side][k][1]][0] - filt_heel[side.opposite()][step_pairs[side][k][1]][0]
                        else:
                            stride_length += (filt_heel[side][step_pairs[side][k][1]][0] * metres_per_pixel[side][step_pairs[side][k][1]]) - \
                                             (filt_heel[side.opposite()][step_pairs[side][k][1]][0] * metres_per_pixel[side][step_pairs[side][k][1]])
                        stride_length = gaitdata.distance_param_sign_convert(stride_length)
                        stride_lengths[side].append(stride_length)
                    stride_duration += (step_pairs[side][k][1] - step_pairs[side][k][0]) / fps
                    stride_durations[side].append(stride_duration)
                    if use_gaitdata:
                        print(str(stride_lengths[side][-1]) + distance_units + ", " + str(stride_durations[side][-1]) + "s")
                    else:
                        print(str(stride_durations[side][-1]) + "s")
                    break

        print("\nStep lengths and/or durations:")
        for k in range(len(step_pairs[side])):
            if use_gaitdata:
                if gaitdata.is_world_units():
                    step_length = filt_heel[side][step_pairs[side][k][1]][0] - \
                                  filt_heel[side.opposite()][step_pairs[side][k][1]][0]
                else:
                    step_length = (filt_heel[side][step_pairs[side][k][1]][0] * metres_per_pixel[side][step_pairs[side][k][1]]) - \
                                  (filt_heel[side.opposite()][step_pairs[side][k][1]][0] * metres_per_pixel[side][
                                      step_pairs[side][k][1]])
                step_length = gaitdata.distance_param_sign_convert(step_length)
                step_lengths[side].append(step_length)  # distance between toes at step heelstrike
            step_durations[side].append((step_pairs[side][k][1] - step_pairs[side][k][
                0]) / fps)  # duration between ipsilateral and contralateral heelstrikes
            if use_gaitdata:
                print(str(step_lengths[side][-1]) + distance_units + ", " + str(step_durations[side][-1]) + "s")
            else:
                print(str(step_durations[side][-1]) + "s")

        if use_gaitdata and sum(step_durations[side]) != 0:
            walking_speed = sum(stride_lengths[side]) / sum(stride_durations[side])
            print("\nWalking speed (stride length / duration): " + str(walking_speed) + distance_units + "/s")

        cadence = len(step_pairs[side]) / sum(step_durations[side])
        print("\nCadence: " + str(cadence * 60) + " steps/min")

        for i in range(len(stride_pairs[side])):

            init_double_supp_durations[side].append([])
            term_double_supp_durations[side].append([])

            print("\nInitial double support time:")
            for k in range(len(hsto_pairs_alt[side])):
                if is_pair_inside_pair(hsto_pairs_alt[side][k], stride_pairs[side][i]) and \
                        not (hsto_pairs_alt[side][k][0] == stride_pairs[side][i][1]):
                    init_double_supp_durations[side][i].append((hsto_pairs_alt[side][k][1] - hsto_pairs_alt[side][k][
                        0]) / fps)  # Initial double support is taken as the time between HS of ipsilateral foot and TO of contralateral foot
                    print(str(init_double_supp_durations[side][i][-1]) + "s")

            print("\nTerminal double support time:")
            for k in range(len(hsto_pairs_alt[side.opposite()])):
                if is_pair_inside_pair(hsto_pairs_alt[side.opposite()][k], stride_pairs[side][i]):
                    term_double_supp_durations[side][i].append(
                        (hsto_pairs_alt[side.opposite()][k][1] - hsto_pairs_alt[side.opposite()][k][0]) / fps)
                    print(str(term_double_supp_durations[side][i][-1]) + "s")

            print("\nTotal double support time:")
            if len(init_double_supp_durations[side][i]) != 1 or len(term_double_supp_durations[side][i]) != 1:
                print("Error? Double support phases are not 1 each.")
                double_support = 'INVALID'
            else:
                print(str(init_double_supp_durations[side][i][0] + term_double_supp_durations[side][i][0]) + "s")
                double_support = init_double_supp_durations[side][i][0] + term_double_supp_durations[side][i][0]

            print("\nFoot off:")
            foot_off = 'INVALID'
            for k in range(len(TO[side])):
                # Search for toe-off of ipsilateral foot during each stride
                if TO[side][k] >= stride_pairs[side][i][0] and TO[side][k] <= stride_pairs[side][i][1]:
                    if i < len(stride_durations[side]):
                      foot_off = ((TO[side][k] - stride_pairs[side][i][0]) / fps) / stride_durations[side][i]
                      foot_off *= 100
                      print(str(foot_off) + "%")
                      break
                    else:
                      print(f"stride duration {side} len {len(stride_durations[side])} is shorter than i={i}")

            print("\nLimp index:")
            if len(init_double_supp_durations[side][i]) != 1 or len(term_double_supp_durations[side][i]) != 1:
                print("Error? Double support phases are not 1 each.")
                limp_index = 'INVALID'
            else:
                for k in range(len(TO[side])):
                    # Search for ipsilateral hsto within cycle
                    if TO[side][k] >= stride_pairs[side][i][0] and TO[side][k] <= stride_pairs[side][i][1]:
                        fc_to_fo = (TO[side][k] - stride_pairs[side][i][0]) / fps # foot contact to foot off time
                        fo_to_fc = (stride_pairs[side][i][1] - TO[side][k]) / fps # foot off to next foot contact time
                        break
                limp_index = fc_to_fo / (fo_to_fc + init_double_supp_durations[side][i][0] + term_double_supp_durations[side][i][0])
                print(str(limp_index))

            print("\nOpposite foot contact:")
            opp_foot_contact = 'INVALID'
            for k in range(len(HS[side.opposite()])):
                # Search for heelstrike of contralateral foot during each stride of ipsilateral foot
                if HS[side.opposite()][k] >= stride_pairs[side][i][0] and HS[side.opposite()][k] <= stride_pairs[side][i][1]:
                    opp_foot_contact = ((HS[side.opposite()][k] - stride_pairs[side][i][0]) / fps) / stride_durations[side][i]
                    opp_foot_contact *= 100
                    print(str(opp_foot_contact) + "%")
                    break

            print("\nOpposite foot off:")
            opp_foot_off = 'INVALID'
            for k in range(len(TO[side.opposite()])):
                # Search for toe-off of contralateral foot during each stride of ipsilateral foot
                if TO[side.opposite()][k] >= stride_pairs[side][i][0] and TO[side.opposite()][k] <= stride_pairs[side][i][1]:
                    opp_foot_off = ((TO[side.opposite()][k] - stride_pairs[side][i][0]) / fps) / stride_durations[side][i]
                    opp_foot_off *= 100
                    print(str(opp_foot_off) + "%")
                    break

            print("\nSingle support time:")
            single_support_time = 'INVALID'
            for k in range(len(tohs_pairs[side.opposite()])):
                if is_pair_inside_pair(tohs_pairs[side.opposite()][k], stride_pairs[side][i]):
                    single_support_time = (tohs_pairs[side.opposite()][k][1] - tohs_pairs[side.opposite()][k][
                        0]) / fps  # Single support is equivalent to time between TO and HS of opposite foot
                    print(str(single_support_time) + "s")

            print("\nStep widths:")
            # Take mediolateral distance between heels at terminal double support
            if use_gaitdata and (gaitdata.framework == "vicon" or gaitdata.framework == "mediapipe" or gaitdata.framework == "mediapipeheavy"):
                for k in range(len(hsto_pairs_alt[side.opposite()])):
                    if is_pair_inside_pair(hsto_pairs_alt[side.opposite()][k], stride_pairs[side][i]):
                        step_width = abs(data[Joint.LEFT_HEEL][hsto_pairs_alt[side.opposite()][k][0]][2] -
                                         data[Joint.RIGHT_HEEL][hsto_pairs_alt[side.opposite()][k][0]][2])
                        print(str(step_width) + distance_units)

            csvwriter.writerow(['Double support (s)', str(double_support)])
            csvwriter.writerow(['Foot off (%)', str(foot_off)])
            csvwriter.writerow(['Limp index', str(limp_index)])
            csvwriter.writerow(['Opposite foot contact (%)', str(opp_foot_contact)])
            csvwriter.writerow(['Opposite foot off (%)', str(opp_foot_off)])
            csvwriter.writerow(['Single support (s)', str(single_support_time)])
            # csvwriter.writerow(['Step width', str(step_width)])
            for k in range(len(step_pairs[side])):
                if is_pair_inside_pair(step_pairs[side][k], stride_pairs[side][i]):
                    if use_gaitdata:
                        csvwriter.writerow(['Step length (m)', str(step_lengths[side][k])])
                    csvwriter.writerow(['Step time (s)', str(step_durations[side][k])])
                    stepDuration = step_durations[side][k]
            if use_gaitdata:
                csvwriter.writerow(['Stride length (m)', str(stride_lengths[side][i])])
            csvwriter.writerow(['Stride time (s)', str(stride_durations[side][i])])
            if use_gaitdata:
                #JSLOW(20230719) - change walking speed to per stride
                # csvwriter.writerow(['Walking speed (m/s)', str(walking_speed)])
                csvwriter.writerow(['Walking speed (m/s)', str(stride_lengths[side][i]/stride_durations[side][i])])
            #JSLOW(20230719) - change cadence to per stride
            #  csvwriter.writerow(['Cadence (step/min)', str(cadence * 60)])
            if stepDuration > 0:
                csvwriter.writerow(['Cadence (step/min)', str(60/stepDuration)])
            else:
                csvwriter.writerow(['Cadence (step/min)', '0'])

        print("\nStep Length Variance: " + str(np.var(step_lengths[side])))
        print("\nStride Length Variance: " + str(np.var(stride_lengths[side])))
        print("\nStep Time Variance: " + str(np.var(step_durations[side])))
        print("\nStride Time Variance: " + str(np.var(stride_durations[side])))

    csvwriter.writerow([''])

    print("\nNON SIDE-SPECIFIC PARAMETERS:")
    print("\n********************")
    cadence = (len(step_pairs[Side.LEFT]) + len(step_pairs[Side.RIGHT])) / (
                sum(step_durations[Side.LEFT]) + sum(step_durations[Side.RIGHT]))
    print("\nAverage Cadence: " + str(cadence * 60) + " steps/min")

    csvwriter.writerow(['Average Cadence', str(cadence * 60)])

    # JSLOW(20230522) - add overriding mid gait cycle pick (base 1 index)
    pick = {Side.LEFT:0, Side.RIGHT:0}
    try: pick[Side.LEFT] = gaitdata.oLeft
    except AttributeError: pass

    try: pick[Side.RIGHT] = gaitdata.oRight
    except AttributeError: pass

    for side in Side:
    # JSLOW(20230522) - add overriding mid gait cycle pick (base 1 index)
        if pick[side] == 0:
            pick[side] = int((len(stride_pairs[side])+1)/2)
        else:
            print(f"INFO: override {side.name} stride pick {pick[side]}")

    if use_gaitdata:
        frameCount = len(data[Joint.LEFT_ANKLE])
    else:
        frameCount = max(HS[Side.LEFT][-1],TO[Side.LEFT][-1],HS[Side.RIGHT][-1],TO[Side.RIGHT][-1]) + 1
    time_axis = np.linspace(0, frameCount / fps, num=frameCount)

    # Output Kinematics
    if use_gaitdata:
        ### Additional kinetic angle plots ###
        # time_axis = np.linspace(0, len(data[Joint.LEFT_ANKLE]) / fps,
        #                         num=len(data[Joint.LEFT_ANKLE]))  # Use time axis from left side for both sides

        knee_flexion = {Side.LEFT: [], Side.RIGHT: []}
        hip_flexion = {Side.LEFT: [], Side.RIGHT: []}
        sole_to_floor = {Side.LEFT: [], Side.RIGHT: []}
        ankle_flexion = {Side.LEFT: [], Side.RIGHT: []}

        line_style = {Side.LEFT: "r", Side.RIGHT: "g"}

# JSLOW(20230216) - add csv output for kinematic data per frame per side
        csvkinetime = {Side.LEFT: open('kinematics_1.csv','w', newline=''),
                       Side.RIGHT: open('kinematics_2.csv','w', newline='')}
        cktwriter = {Side.LEFT: csv.writer(csvkinetime[Side.LEFT], lineterminator='\n'), 
                    Side.RIGHT: csv.writer(csvkinetime[Side.RIGHT], lineterminator='\n')}

        kinedata['ankle flexion'], kinedata['knee flexion'], kinedata['hip flexion'] = {}, {}, {}
 
        for side in Side:
            # JSLOW(20230522) - add overriding mid gait cycle pick (base 1 index)
            # if pick[side] == 0:
            #     pick[side] = int((len(stride_pairs[side])+1)/2)

            for i in range(len(time_axis)):
                knee_flexion[side].append(find_flexion_deg(gaitdata.walk_direction, filt_knee[side][i], filt_hip[side][i], filt_knee[side][i], filt_ank[side][i]))
                if gaitdata.framework != 'vicon':
                    hip_flexion[side].append(-find_flexion_deg(gaitdata.walk_direction, filt_midhip[i], filt_midshoulder[i], filt_hip[side][i], filt_knee[side][i]))

                sole_vector = filt_toe[side][i] - filt_heel[side][i]
                sole_sine = sole_vector[1] / np.linalg.norm(sole_vector)
                sole_to_floor[side].append(np.rad2deg(np.arcsin(sole_sine)))

                ankle_flexion[side].append(
                    find_ankleDpFlex_deg(filt_knee[side][i], filt_ank[side][i], filt_heel[side][i], filt_toe[side][i]))
            # for k in HS[side]:
            #     print(side.name + " HS sole angles")
            #     print(sole_to_floor[side][k])

            cktwriter[side].writerow(["time","ankle_flexion", "knee_flexion", "hip_flexion"])
            flexions = np.array([time_axis, ankle_flexion[side], knee_flexion[side], hip_flexion[side]]).T
            cktwriter[side].writerows(flexions)

            kinedata['ankle flexion'][side] = np.array(ankle_flexion[side])
            kinedata['knee flexion'][side] = np.array(knee_flexion[side])
            kinedata['hip flexion'][side] = np.array(hip_flexion[side])
 
            for i in range(len(stride_pairs[side])):

                print(side.name+" Stride "+str(i))

                start, end = stride_pairs[side][i][0], stride_pairs[side][i][1]

                print("\nKnee flexion/extension")

                knee_peaks, _ = scipy.signal.find_peaks(knee_flexion[side][start:end], width=3)
                knee_troughs, _ = scipy.signal.find_peaks(-np.array(knee_flexion[side][start:end]), width=3)

                temp_title = gaitdata.framework + " " + side.name + " Knee Flexion " + str(i+1)
                plt.figure()
                plt.gcf().canvas.manager.set_window_title(temp_title)
                plt.plot(time_axis[start:end], knee_flexion[side][start:end], line_style[side])
                plt.plot(np.array(time_axis[start:end])[knee_peaks], np.array(knee_flexion[side][start:end])[knee_peaks], "x")
                plt.plot(np.array(time_axis[start:end])[knee_troughs], np.array(knee_flexion[side][start:end])[knee_troughs], "x")
                for k, x in enumerate(knee_peaks):
                    plt.gca().annotate("Peak", (np.array(time_axis[start:end])[x], np.array(knee_flexion[side][start:end])[x]))
                    print("Peak of "+str(np.array(knee_flexion[side][start:end])[x])+" at "+str((x / (end-start))*100)+"%")
                for k, x in enumerate(knee_troughs):
                    plt.gca().annotate("Trough", (np.array(time_axis[start:end])[x], np.array(knee_flexion[side][start:end])[x]))
                    print("Trough of " + str(np.array(knee_flexion[side][start:end])[x]) + " at " + str(
                        (x / (end - start)) * 100) + "%")
                print("Final extension of "+str(np.array(knee_flexion[side][start:end])[-1]))
                plt.gca().set(ylabel=side.name+" Knee Flexion (deg)")
                plt.ylim([0,60])
                plt.gca().set(xlabel="Percent of cycle (%)")
                plt.xticks(np.linspace(time_axis[start], time_axis[end-1], num=11), np.linspace(0, 100, num=11).astype(int))
                plt.gcf().savefig(fig_dir + temp_title)
                plt.close()

                if gaitdata.framework != 'vicon':
                    temp_title = gaitdata.framework + " " + side.name + " Hip Flexion " + str(i+1)
                    plt.figure()
                    plt.gcf().canvas.manager.set_window_title(temp_title)
                    plt.plot(time_axis[start:end], hip_flexion[side][start:end], line_style[side])
                    plt.gca().set(ylabel=side.name + " Hip Flexion (deg)")
                    plt.gca().set(xlabel="Percent of cycle (%)")
                    plt.xticks(np.linspace(time_axis[start], time_axis[end - 1], num=11), np.linspace(0, 100, num=11).astype(int))
                    plt.gcf().savefig(fig_dir + temp_title)
                    plt.close()

                print("\nAnkle Dorsi/Plantar flexion")

                ankle_peaks, _ = scipy.signal.find_peaks(ankle_flexion[side][start:end], width=3)
                ankle_troughs, _ = scipy.signal.find_peaks(-np.array(ankle_flexion[side][start:end]), width=3)

                temp_title = gaitdata.framework + " " + side.name + " Ankle DorsiPlantarflexion" + str(i+1)
                plt.figure()
                plt.gcf().canvas.manager.set_window_title(temp_title)
                plt.plot(time_axis[start:end], ankle_flexion[side][start:end], line_style[side])
                plt.plot(np.array(time_axis[start:end])[ankle_peaks], np.array(ankle_flexion[side][start:end])[ankle_peaks], "x")
                plt.plot(np.array(time_axis[start:end])[ankle_troughs], np.array(ankle_flexion[side][start:end])[ankle_troughs], "x")
                for k, x in enumerate(ankle_peaks):
                    plt.gca().annotate("Peak", (np.array(time_axis[start:end])[x], np.array(ankle_flexion[side][start:end])[x]))
                    print("Peak of "+str(np.array(ankle_flexion[side][start:end])[x])+" at "+str((x / (end-start))*100)+"%")
                for k, x in enumerate(ankle_troughs):
                    plt.gca().annotate("Trough", (np.array(time_axis[start:end])[x], np.array(ankle_flexion[side][start:end])[x]))
                    print("Trough of " + str(np.array(ankle_flexion[side][start:end])[x]) + " at " + str(
                        (x / (end - start)) * 100) + "%")
                print("First HS value: " + str(np.array(knee_flexion[side][start:end])[0]))
                print("Second HS value: " + str(np.array(knee_flexion[side][start:end])[-1]))
                plt.gca().set(ylabel=side.name + " Ankle Dorsi/Plantarflexion (deg)")
                plt.gca().set(xlabel="Percent of cycle (%)")
                plt.xticks(np.linspace(time_axis[start], time_axis[end - 1], num=11), np.linspace(0, 100, num=11).astype(int))
                plt.gcf().savefig(fig_dir + temp_title)
                plt.close()

    # JSLOW(20230208) - add csv output for min/max kinematics data per stride per side
    # JSLOW (5 Feb 2023): output min/max kinematics value per stride
    # JSLOW (10 May 2023): output the midpoint stride. i is base 0, thus + 1 to be base 1 index
    csvkinematics = open('kinematics.csv', 'w', newline='')
    ckwriter = csv.writer(csvkinematics, lineterminator='\n')
    for side in Side:
        ckwriter.writerow([side.name])
        i = pick[side]-1  # convert from base 1 to base 0
        start, end = stride_pairs[side][i][0], stride_pairs[side][i][1]
        for kineType, kineAngles in kinedata.items():
            ckwriter.writerow([f"Max {kineType}(deg)", str(np.max(kineAngles[side][start:end]))])
            ckwriter.writerow([f"Min {kineType}(deg)", str(np.min(kineAngles[side][start:end]))])
        ckwriter.writerow(['Mid-stride time', f'{time_axis[start]} {time_axis[end]}'])
        ckwriter.writerow(['Heel Strike', " ".join(f'{time_axis[x]}' for x in HS[side])])
        ckwriter.writerow(['Toe Off', " ".join(f'{time_axis[x]}' for x in TO[side])])

# JSLOW(20230529) take 4 points to define 2 vectors
def find_flexion_deg(face_direction: Side, upperFrom, upperTo, lowerFrom, lowerTo):
# def find_flexion_deg(face_direction: Side, upper_joint, middle_joint, lower_joint):

    # Assuming subject is facing postive x direction (right),
    # Using right-handed coordinate axes,
    # Clockwise rotation will be computed as flexion (positive),
    # Anticlockwise as extension (negative)

    vector_upper = upperTo[0:2] - upperFrom[0:2]
    vector_lower = lowerTo[0:2] - lowerFrom[0:2]

    # Deprecated method which only yields positive angles even during joint extension.
    # cosine = np.dot(vector_lower, vector_upper) / (np.linalg.norm(vector_lower) * np.linalg.norm(vector_upper))
    # angle_rad = np.arccos(cosine)
    # angle_rad = np.pi - angle_rad

    cross = np.cross(vector_upper, vector_lower)
    sine = np.linalg.norm(cross) / (np.linalg.norm(vector_lower) * np.linalg.norm(vector_upper))
    angle_rad = np.arcsin(sine)

    if (face_direction == Side.RIGHT and cross < 0) or \
            (face_direction == Side.LEFT and cross > 0):
        # z-component is negative, extension
        angle_rad = -angle_rad

    return np.rad2deg(angle_rad)

def find_ankleDpFlex_deg(knee, ankle, heel, toe):
    vector_shank = knee - ankle
    vector_foot = toe - heel
    cosine = np.dot(vector_shank, vector_foot) / (np.linalg.norm(vector_shank) * np.linalg.norm(vector_foot))
    angle_rad = np.arccos(cosine)
    return -(np.rad2deg(angle_rad) - 90)

def simple_moving_average(array, window_size):
    if window_size%2 == 0:
        raise ValueError("Window size must be odd integer.")

    half_size = int((window_size - 1) / 2)

    new_array = []
    for i in range(half_size, (len(array) - half_size)):
        moving_average = 0
        for k in range(-half_size, half_size+1):
            moving_average += array[i+k]
        moving_average /= window_size
        new_array.append(moving_average)

    print(len(array))
    print(len(new_array))

    return new_array

def exp_moving_average(array, alpha):
    new_array = [array[0]]
    for i in range(1, len(array)):
        new_entry = alpha*array[i] + (1-alpha)*new_array[i-1]
        new_array.append(new_entry)
    return new_array