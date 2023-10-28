import pandas as pd
import pickle
import json

print("Hello World")

filepath = 'C:\\Users\\whyxi\\OneDrive\\Pictures\\Backup\\NUS\\NUS Academics\\FYP EE4002D\\Gait Analysis\\Code\\gait-lyr-repo\\pickles\\H001\\Free walk\\AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR.pkl'
filepath2 = 'C:\\Users\\whyxi\\OneDrive\\Pictures\\Backup\\NUS\\NUS Academics\\FYP EE4002D\\Gait Analysis\\Code\\gait-lyr-repo\\pickles\\H002\\Free walk(Left)\\AR Gait_P_H002_Free walk_21-06-2022_15-24-24_noAR.pkl'
filepath3 = 'C:/Users/whyxi/OneDrive/Pictures/Backup/NUS/NUS Academics/FYP EE4002D/Gait Analysis/Code/gait-lyr-repo/pickles/H001/Free walk/AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR.pkl'
# PROBLEM filepath4 = 'C:\\Users\whyxi\OneDrive\Pictures\Backup\NUS\NUS Academics\FYP EE4002D\Gait Analysis\Code\gait-lyr-repo\pickles\H001\Free walk\AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR.pkl'
filepath5 = '..\pickles\H001\Free walk\AR Gait_P_H001_Free walk_14-06-2022_16-25-45_noAR.pkl'
# filepath6 = r"{}".format(filepath4)

student_names = {
  'Student 1': {
        'Name': "Alice", 'Age' :10, 'Grade':4,
    },   
    'Student 2': {
        'Name':'Bob', 'Age':11, 'Grade':5
    },   
    'Student 3': {
        'Name':'Elena', 'Age':14, 'Grade':8
    }   
}


# with open(filepath, 'wb') as f:  # open a text file
   # pickle.dump(student_names, f) # serialize the list

with open(filepath2, 'rb') as f:
    student_names_loaded = pickle.load(f) # deserialize using load()
    print(student_names_loaded.data) # print student names

# Serializing json
json_object = json.dumps(student_names_loaded.data, indent=4)
 
# Writing to sample.json
with open("qqwweerr.json", "w") as outfile:
    outfile.write(json_object)

# df = pd.read_pickle('student_file.pkl')
# print(df)