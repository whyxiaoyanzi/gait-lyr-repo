import cv2
import os
import time
import numpy as np
import gaitevents
import pyopenpose as op

def process(walk_direction: gaitevents.Side, raw_video_file: str, save_dir: str = ".", patientinfo = dict(), rotate_cw=False):

    poseModel = op.PoseModel.BODY_25B
    print(op.getPoseBodyPartMapping(poseModel))
    # print(op.getPoseNumberBodyParts(poseModel))
    # print(op.getPosePartPairs(poseModel))
    # print(op.getPoseMapIndex(poseModel))
    poseName2Num = {v:k for k,v in op.getPoseBodyPartMapping(poseModel).items()}

    try:
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        params = {
            "model_folder": os.environ['MODEL_DIR'],
            "model_pose"  : "BODY_25B"
        }
        
        video_capture = cv2.VideoCapture(raw_video_file)
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_width = video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        video_height = video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print("Video info:")
        print("FPS: "+str(fps)+" Width: "+str(video_width)+" Height: "+str(video_height))
        if fps < 1:
            raise ValueError(f"Bad video file: {raw_video_file}")

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        #video_writer = cv2.VideoWriter(save_dir+"/openpose_0.avi", fourcc, round(fps), (int(video_capture.get(3)), int(video_capture.get(4))))
        out_video_file = os.path.splitext(os.path.basename(raw_video_file))[0] + ".avi"
        video_writer = cv2.VideoWriter(save_dir+"/"+out_video_file, fourcc, round(fps), (int(video_width), int(video_height)))

        net_res_longerside = "448" # 4gb vram, scale_number 1
        # net_res_longerside = "544" # 10gb vram, scale_number 4
        # net_res_longerside = "832" # "max" accuracy, scale_number 4, portrait (768x1024)

        if video_width > video_height:
            # landscape video detected
            params["net_resolution"] = net_res_longerside+"x-1"
        else:
            # portrait video detected
            params["net_resolution"] = "-1x"+net_res_longerside
        if rotate_cw:
            params["frame_rotate"] = 90

        # params["scale_number"] = "4"
        # params["scale_gap"] = "0.25"
        posescore_thres = 0.0

        keypoint_lists = {'LHip':[], 'RHip':[], 'LKnee':[], 'RKnee':[], 'LHeel':[], 'LBigToe':[], 'LSmallToe':[], 'RHeel':[], 'RBigToe':[], 'RSmallToe':[], 'LAnkle':[], 'RAnkle':[], 'LShoulder':[], 'RShoulder':[]}

        # Starting OpenPose
        print(f"Openpose parameters: {params}")
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()

        time_start = time.time()
        frame_no = 0
        while video_capture.isOpened():
            # Process Image
            ret, frame = video_capture.read()
            # time_sf = time.time()
            if not ret:
                if frame_no==0:
                    print("Can't receive frame (stream end?). Exiting ...")
                else:
                    print(frameAt+"Last frame processed")
                break
            datum = op.Datum()
            datum.cvInputData = frame
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            # Display Image
            #print("Body keypoints: \n" + str(datum.poseKeypoints))
            output_frame = datum.cvOutputData
            # cv2.putText(output_frame, 'Left heel: '+str(datum.poseKeypoints[0][21]), (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1,
            #             cv2.LINE_AA)
            # cv2.putText(output_frame, 'Left hip: ' + str(datum.poseKeypoints[0][11]), (20, 40), cv2.FONT_HERSHEY_PLAIN,
            #             1, (255, 255, 255), 1,
            #             cv2.LINE_AA)
            # cv2.putText(output_frame, 'Left heel adj: ' + str(datum.poseKeypoints[0][21]- datum.poseKeypoints[0][11]), (20, 60), cv2.FONT_HERSHEY_PLAIN,
            #             1, (255, 255, 255), 1,
            #             cv2.LINE_AA)
            # cv2.putText(output_frame, 'Process fps: '+str(1/(time.time()-time_sf)), (20, 20), cv2.FONT_HERSHEY_PLAIN,
            #             1, (255, 255, 255), 1,
            #             cv2.LINE_AA)
            #cv2.imshow("OpenPose 1.7.0 - " + raw_video_file, output_frame)
            # video_writer.write(output_frame)
            frameAt = f"Frame {frame_no}|"
            if datum.poseKeypoints is not None and datum.poseScores[np.argmax(datum.poseScores)] > posescore_thres:
                posescore_max = np.argmax(datum.poseScores)
                if np.size(datum.poseScores) != 1:
                    print(frameAt+"Posescores: " + str(datum.poseScores))
                    print(frameAt+"Posescore idx used: " + str(posescore_max))
                    kp_num = poseName2Num['RHip']
                    textpos = (int(datum.poseKeypoints[posescore_max][kp_num][0]), int(datum.poseKeypoints[posescore_max][kp_num][1]))
                    cv2.putText(output_frame, 'X', textpos, cv2.FONT_HERSHEY_PLAIN,
                        1, (255, 255, 255))
                for keypoint, list in keypoint_lists.items():
                    kp_num = poseName2Num[keypoint]
                    if datum.poseKeypoints[posescore_max][kp_num][0] == 0:
                        if len(list) > 0:
                            print(frameAt+"Keypoint "+keypoint+" not found. Using previous values.")
                            list.append(list[-1])
                        else:
                            list.append(np.array([np.nan, np.nan]))
                    else:
                        list.append(datum.poseKeypoints[posescore_max][kp_num])
                        list[-1][1] = -list[-1][1] + int(video_height) # flip and translate y-values
            else:
                print(frameAt+"No Person detected. Using previous values or nan if no valid previous value.")
                for keypoint, list in keypoint_lists.items():
                    if len(list) > 0:
                        list.append(list[-1])
                    else:
                        list.append(np.array([np.nan, np.nan]))
            video_writer.write(output_frame)
            frame_no = frame_no +1

        cv2.destroyAllWindows()

        print("Openpose processing complete. Time taken: "+str(time.time() - time_start)+"s. Average frames processed per second: " + str(
            video_capture.get(cv2.CAP_PROP_FRAME_COUNT) / (time.time() - time_start))+".")

        video_capture.release()
        video_writer.release()

        # Replace nan values at start, if any, with earliest available value
        for keypoint, list in keypoint_lists.items():
            if np.isnan(list[0][0]):
                keypoint_appeared = False
                for i in range(len(list)):
                    if not np.isnan(list[i][0]):
                        replace_up_to = i
                        print("Keypoint "+keypoint+" missing "+str(replace_up_to)+" frames from start. Replacing with earliest available value.")
                        keypoint_appeared = True
                        break
                    if i == len(list)-1:
                        print("Warning: Keypoint "+keypoint+" not detected for entire video.")
                        break
                if keypoint_appeared:
                    for i in range(replace_up_to):
                        list[i] = list[replace_up_to]

        midhip = 0.5 * (np.array(keypoint_lists['RHip'])[:,0:2] + np.array(keypoint_lists['LHip'])[:,0:2])
        mid_shoulder = 0.5 * (np.array(keypoint_lists['RShoulder'])[:,0:2] + np.array(keypoint_lists['LShoulder'])[:,0:2])
        mid_left_toes = 0.5 * (np.array(keypoint_lists['LSmallToe'])[:,0:2] + np.array(keypoint_lists['LBigToe'])[:,0:2])
        mid_right_toes = 0.5 * (np.array(keypoint_lists['RSmallToe'])[:, 0:2] + np.array(keypoint_lists['RBigToe'])[:, 0:2])

        #save_filename = 'gaitdata_openpose.pkl'
        save_filename = os.path.splitext(os.path.basename(raw_video_file))[0] + ".pkl"

        gaitdata = gaitevents.GaitData("openpose", fps)

        gaitdata.walk_direction = walk_direction
        gaitdata.patientinfo = patientinfo

        gaitdata.data[gaitevents.Joint.LEFT_HEEL] = np.array(keypoint_lists['LHeel'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_HEEL] = np.array(keypoint_lists['RHeel'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.LEFT_TOE_BIG] = np.array(keypoint_lists['LBigToe'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_TOE_BIG] = np.array(keypoint_lists['RBigToe'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.LEFT_TOE_SMALL] = np.array(keypoint_lists['LSmallToe'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_TOE_SMALL] = np.array(keypoint_lists['RSmallToe'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.LEFT_ANKLE] = np.array(keypoint_lists['LAnkle'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_ANKLE] = np.array(keypoint_lists['RAnkle'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.LEFT_KNEE] = np.array(keypoint_lists['LKnee'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_KNEE] = np.array(keypoint_lists['RKnee'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.LEFT_HIP] = np.array(keypoint_lists['LHip'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.RIGHT_HIP] = np.array(keypoint_lists['RHip'])[:, 0:2]
        gaitdata.data[gaitevents.Joint.MIDDLE_HIP] = midhip
        gaitdata.data[gaitevents.Joint.MID_SHOULDER] = mid_shoulder
        gaitdata.data[gaitevents.Joint.LEFT_TOES] = mid_left_toes
        gaitdata.data[gaitevents.Joint.RIGHT_TOES] = mid_right_toes
        gaitevents.save_object(gaitdata, save_dir+"/"+save_filename)

        print("Data saved to "+save_dir+"/"+save_filename)

    except Exception as e:
        raise e
