import os
import gait
import pandas as pd

data = pd.read_csv('C:/Users/chong/Downloads/OneDrive_1_9-2-2023/gaitanalysisvideo-main)/gaitanalysisvideo-main/Vicon HS Start End.csv', header = None) 
file = data.iloc[0,1:2]
left_start = data.iloc[1,1:]
left_end = data.iloc[2,1:]
right_start = data.iloc[4,1:]
right_end = data.iloc[5,1:]
lt_start = data.iloc[9,1:]
lt_end = data.iloc[10,1:]
rt_start = data.iloc[12,1:]
rt_end = data.iloc[13,1:]
all_timings = {}
for i in range(1, len(file)+1):
    all_timings[file[i]] = [[[left_start[i],left_end[i]],[right_start[i],right_end[i]]],[[lt_start[i],lt_end[i]],[rt_start[i],rt_end[i]]]]

def process_files(directory_path):
    file_paths = {}
    for root, dirs, files in os.walk(directory_path, topdown=True):
        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for _, _, filenames in os.walk(subdir_path, topdown=True):
                for filename in filenames:
                    file_name, file_extension = os.path.splitext(filename)
                    file_paths[file_name] = os.path.join(subdir_path, filename)
    return file_paths
file_list = process_files("C:\\Users\\chong\\Downloads\\pickles")

for i in file:
    #gait.analyse_cbta(file_list[i], "C:\\Users\\chong\\Desktop\\pkl", 0.8,0.8, i, all_timings)
    gait.analyse_cbta("C:\\Users\\chong\\Downloads\\pickles_check\\H001\\Free walk\\AR Gait_P_H001_Free walk_14-06-2022_16-26-15_noAR.pkl", "C:\\Users\\chong\\Desktop\\pkl", 0.8,0.8, i, all_timings)
    
