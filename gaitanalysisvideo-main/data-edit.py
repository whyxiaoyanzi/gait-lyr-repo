import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

all_valid_datasets = {
  "sheet_name": [],
  "argmin_cbta_time": [],
  "argmax_cbta_time": [],
  "argmin_vicon_time": [], 
  "argmax_vicon_time": [],
  "min_cbta_knee_flexion_angle":[],
  "max_cbta_knee_flexion_angle":[],
  "min_vicon_knee_flexion_angle":[],
  "max_vicon_knee_flexion_angle":[]
}

# all_valid_datasets["argmax_cbta_time"].append("alalakala")
# print("hihi")
# print(all_valid_datasets["argmax_cbta_time"][3])

inputfilename = 'All_Kinematics_Left.xlsx'
xl = pd.ExcelFile(inputfilename)
sheet_names = np.array(xl.sheet_names)
# print(sheet_names[2:5])
# print(len(sheet_names))

for i in range(len(sheet_names)):
# for i in range(1):
    df = pd.read_excel(inputfilename, sheet_name=i)
    # print(df)

    number_cbta_readings = len(df.iloc[0]) - 1
    print('number_cbta_readings')
    print(number_cbta_readings)   

    cbta_time = np.array(df.iloc[0][1:(number_cbta_readings+1)].to_numpy(), dtype=float)
    cbta_knee_flexion_angle = np.array(df.iloc[2][1:(number_cbta_readings+1)].to_numpy(), dtype=float)
    sine_cbta_knee_flexion_angle = np.sin(cbta_knee_flexion_angle * np.pi / 180) * 100
    cbta_confidence = np.array(df.iloc[4][1:(number_cbta_readings+1)].to_numpy(), dtype=float)

    number_vicon_readings = 51
    vicon_time = np.array(df.iloc[7][1:(number_vicon_readings+1)].to_numpy(), dtype=float)
    vicon_knee_flexion_angle = np.array(df.iloc[9][1:(number_vicon_readings+1)].to_numpy(), dtype=float)
    sine_vicon_knee_flexion_angle = np.sin(vicon_knee_flexion_angle * np.pi / 180) * 100
    
    if i==5:
        degree = 15
        figure, axis = plt.subplots(2, 1) #2 rows, 1 column
        vicon_coefficients = np.polyfit(vicon_time, sine_vicon_knee_flexion_angle, degree)
        vicon_poly_function = np.poly1d(vicon_coefficients)
        axis[0].plot(vicon_time, vicon_knee_flexion_angle, 'o', vicon_time, vicon_poly_function(vicon_time), ls = '-', color = 'b')
        
        gold_standard_sine_cbta_knee_flexion_angle = vicon_poly_function(cbta_time)
        
        cbta_coefficients = np.polyfit(cbta_time, sine_cbta_knee_flexion_angle, degree)
        cbta_poly_function = np.poly1d(cbta_coefficients)
        axis[1].plot(cbta_time, cbta_knee_flexion_angle, 'o', cbta_time, cbta_poly_function(cbta_time), ls = ':', color = 'r')
        plt.show()

    '''
    print('cbta_time')
    print(cbta_time)
    print('cbta_knee_flexion_angle')
    print(cbta_knee_flexion_angle)
    print('sine_cbta_knee_flexion_angle')
    print(sine_cbta_knee_flexion_angle)

    print('vicon_time')
    print(vicon_time)
    print('vicon_knee_flexion_angle')
    print(vicon_knee_flexion_angle)
    print('sine_vicon_knee_flexion_angle')
    print(sine_vicon_knee_flexion_angle)

    print('np.mean(cbta_confidence)')
    print(np.mean(cbta_confidence))
    '''

    number_of_invalid_vicon = 0
    for x in vicon_knee_flexion_angle:
        if np.isnan(x):
            number_of_invalid_vicon += 1
    print('number_of_invalid_vicon')
    print(number_of_invalid_vicon)

    # cbta's readings have high confidence AND vicon has readings >75% of the time
    if (np.mean(cbta_confidence) > 0.72 ) and (number_of_invalid_vicon < 0.23*number_vicon_readings):
        is_valid_dataset = True
    else:
        is_valid_dataset = False
    # print(is_valid_dataset)

    if is_valid_dataset == True:
        all_valid_datasets["sheet_name"].append(sheet_names[i])

        min_cbta_knee_flexion_angle_time = float(cbta_time[np.argmin(cbta_knee_flexion_angle)])
        all_valid_datasets["argmin_cbta_time"].append(min_cbta_knee_flexion_angle_time)
        max_cbta_knee_flexion_angle_time = float(cbta_time[np.argmax(cbta_knee_flexion_angle)])
        all_valid_datasets["argmax_cbta_time"].append(max_cbta_knee_flexion_angle_time)

        min_vicon_knee_flexion_angle_time = float(vicon_time[np.argmin(vicon_knee_flexion_angle)])
        all_valid_datasets["argmin_vicon_time"].append(min_vicon_knee_flexion_angle_time)
        max_vicon_knee_flexion_angle_time = float(vicon_time[np.argmax(vicon_knee_flexion_angle)])
        all_valid_datasets["argmax_vicon_time"].append(max_vicon_knee_flexion_angle_time)

        min_cbta_knee_flexion_angle = float(np.min(cbta_knee_flexion_angle))
        all_valid_datasets["min_cbta_knee_flexion_angle"].append(min_cbta_knee_flexion_angle)
        max_cbta_knee_flexion_angle = float(np.max(cbta_knee_flexion_angle))
        all_valid_datasets["max_cbta_knee_flexion_angle"].append(max_cbta_knee_flexion_angle)

        min_vicon_knee_flexion_angle = float(np.min(vicon_knee_flexion_angle))
        all_valid_datasets["min_vicon_knee_flexion_angle"].append(min_vicon_knee_flexion_angle)
        max_vicon_knee_flexion_angle = float(np.max(vicon_knee_flexion_angle))
        all_valid_datasets["max_vicon_knee_flexion_angle"].append(max_vicon_knee_flexion_angle)



        '''
        length_vicon = len(df.iloc[2])
        print(length_vicon)
        length_vicon_time = len(df.iloc[2])
        print(length_vicon)
        col_length = len(df['CBTA'])
        print(col_length)
        '''

df2 = pd.DataFrame.from_dict(all_valid_datasets) 
df2.to_excel ('comparison_left.xlsx', index=False, header=True)