import platform
import pandas as pd
import json, csv
import pickle
from datetime import datetime
from operator import itemgetter
from pprint import pprint
import importlib
import itertools
from copy import deepcopy
import numpy as np
import math, time, collections, os, errno, sys, code, random
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import shutil
import configparser
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import spatial



def maximum_subset(list_1, list_2):
    result = []
    for element in list_1:
        if element in list_2:
            result.append(element)
    return result

def remove_flat(pds):
    attrList = list(pds)
    flat = []
    for attribute in attrList:
        if attribute not in ["EQP_ID", "START_TIME", "END_TIME", "LOT_ID", "RECIPE", \
                             "RUN_ID", "STEP_NUM", "LAYER", "TRACE_DATE", "TRACE_SEC"]:
            tempSeries = pds[attribute]
            tempMean = tempSeries.mean()
            isFlat = True
            for val in tempSeries:
                if tempMean != val:
                    isFlat = False
                    break
            if isFlat:
                flat.append(attribute)
    return pds[[at for at in attrList if at not in flat]]



def load_data_index(data_folder_path, tool_id, start_time, end_time, input_mode):
    
    print("@@@@@@@@@@@@ load_data_index start! @@@@@@@@@@@@")
    
    ### load data_index
    
    if input_mode == "ALL":
        
        ### load data_index from ALL folder (.pickle)    

        with open(os.path.join(data_folder_path, "ALL", tool_id + ".pickle"), 'rb') as file:
            print("Load file: " + os.path.join(data_folder_path, "ALL", tool_id + ".pickle"))
            data_index = pickle.load(file)
            
            
        ### only reserve PDS and time, and add LOT_ID
        ### remove data that length is less than 20000
        
        PDS_path = os.path.join(data_folder_path, "PDS")
        
        data_index = [{"PDS": lot["PDS"], \
                       "LOT_ID": lot["PDS"][ : lot["PDS"].find("#")], \
                       "time": lot["time"]} \
                       for lot in data_index if len(pd.read_csv(open(os.path.join(PDS_path, lot["PDS"])))) >= 20000]
                                            
    elif input_mode == "PDS":
        
        ### load PDS_index from PDS folder
        
        PDS_index_path = os.path.join(data_folder_path, "PDS", "PDS_index.json")
        with open(PDS_index_path, "r") as input_file:
            PDS_index = json.load(input_file)["valid"]
            
        ### get data_index (we need "PDS", "time" and "LOT_ID")
        
        PDS_path = os.path.join(data_folder_path, "PDS")
        
        data_index = []
        for element in PDS_index:
            ### change lot_id to LOT_ID
            element["LOT_ID"] = element.pop("lot_id")
            ### get run_id
            run_id = element["LOT_ID"][:2] + "_" + element["LOT_ID"]
            
            if element["LOT_ID"][:2] == tool_id:
                for file_name in os.listdir(PDS_path):
                    if file_name.find(run_id) != -1:
                        ### find the file
                        element["PDS"] = file_name
                
                data_index.append(element)
                
    
    ### change time's datatype from string to datetime
    ### example: 2018-01-02 01:40:59
    for lot_index, lot in enumerate(data_index):
        # print(str(lot) + " " + data_index[lot_index]["time"])
        data_index[lot_index]["time"] = datetime.strptime(data_index[lot_index]["time"], "%Y-%m-%d %H:%M:%S")
    
    
    ### sort by time
    
    data_index = sorted(data_index, key = itemgetter("time")) 
    
    
    ### check start_time and end_time
    
    if start_time != None and end_time != None:
        
        start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    
        data_index = [element for element in data_index if \
                      (element["time"] > start_time and element["time"] < end_time)]
    
    
    print("@@@@@@@@@@@@ load_data_index end!   @@@@@@@@@@@@")
    
    return data_index



def integrate_attribute(data_folder_path, tool_id, data_index):
    
    print("@@@@@@@@@@@@ integrate_attribute start! @@@@@@@@@@@@")
    
    
    ### integrate attributes for all PDS (remove_flat first!)
    
    PDS_path = os.path.join(data_folder_path, "PDS")
    
    attribute_list = []
    for lot in data_index:
        df = pd.read_csv(open(os.path.join(PDS_path, lot["PDS"])))
        #print("b: " + str(len(list(df))))
        df = remove_flat(df)
        #print("a: " + str(len(list(df))))
        attribute_list.append(list(df))
    
    unique_attribute = {}
    for attribute in attribute_list:
        ### initialize unique_attribute
        if list(unique_attribute) == []:
            unique_attribute[len(attribute)] = [attribute]
            continue
        
        ### check if attribute is in unique_attribute or not
        for unique_attribute_in_specific_length in list(unique_attribute.values()):
            
            if attribute not in unique_attribute_in_specific_length:
                ### unique_attribute didn't have such a key, you need to declare a list
                if len(attribute) not in unique_attribute:
                    unique_attribute[len(attribute)] = []
                
                unique_attribute[len(attribute)].append(attribute)
                
        #print(len(unique_attribute), len(attribute))
    
    
    ### check  unique_attribute
    
    _maximum_subset = []
    
    if len(unique_attribute) != 1:
        print(tool_id + " has multiple combinations of attributes!")
        
        ### get the maximun_subset of all combinations of attributes
        
        unique_attribute_list = []
        for unique_attribute_in_specific_length in list(unique_attribute.values()):
            for element in unique_attribute_in_specific_length:
                unique_attribute_list.append(element)
        
        _maximum_subset = unique_attribute_list[0]
        for i in range(len(unique_attribute_list) - 1):
            _maximum_subset = maximum_subset(_maximum_subset, unique_attribute_list[i + 1])
            
    else:
        print(tool_id + " has a unique combination of attributes!")
        
        ### get the maximun_subset of all combinations of attributes
        
        _maximum_subset = list(unique_attribute.values())[0][0]
    
    
    print("@@@@@@@@@@@@ integrate_attribute end!   @@@@@@@@@@@@")
    
    return _maximum_subset



def choose_important_time(data_folder_path, tool_id, data_index):

    print("@@@@@@@@@@@@ choose_important_time start! @@@@@@@@@@@@")
    
    PDS_path = os.path.join(data_folder_path, "PDS")
    
    time_index = {}
    
    for lot_idx, lot in enumerate(data_index):
        
        df = pd.read_csv(open(os.path.join(PDS_path, lot["PDS"])))
    
        ### get the time attribute from three files
        files = ['n', 'mqw', 'p']
        start_end = {'n': {'start': -1, 'end': -1}, 'mqw': {'start': -1, 'end': -1}, 'p': {'start': -1, 'end': -1}}
        for file_name in files:
            targets = []
            with open(os.path.join(data_folder_path, "others", "LAYER", file_name + ".csv")) as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    targets.append(row[0].replace(' ', ''))
            front = []
            last = []
            for t in targets:
                temp = df[df['LAYER'].str.replace(' ', '').str.contains(t, na=False)]
                if not temp.empty:
                    front.append(temp.index[0])
                    last.append(temp.index[-1])
        
            front.sort()
            last.sort(reverse=True)
            #print(file_name + ' range: ' + str(front[0]) + ' ~ ' + str(last[0]))
            start_end[file_name]['start'] = front[0]
            start_end[file_name]['end'] = last[0]
    
        #print(str(start_end['n']['start']) + ' ~ ' + str(start_end['mqw']['end']) + ' ~ ' + str(start_end['p']['end']))
        #print('----------------------------------------------------')
        
        time_index[lot["LOT_ID"]] = {}
        time_index[lot["LOT_ID"]]["start"] = int(start_end['n']['start'])
        time_index[lot["LOT_ID"]]["end"] = int(start_end['p']['end'])
    
    
    print("@@@@@@@@@@@@ choose_important_time end! @@@@@@@@@@@@")
    #print(time_index)
    return time_index
    

    
def choose_important_sensor(data_folder_path, tool_id, data_index, attribute):
    
    print("@@@@@@@@@@@@ choose_important_sensor start! @@@@@@@@@@@@")
    
    ### This file must contain 10 discrete data's attribute! Please remember to add them into your file.
    
    if os.path.isfile( os.path.join(data_folder_path, "others", "SENSOR", tool_id + ".csv") ):
    
        important_sensor = []
        
        with open( os.path.join(data_folder_path, "others", "SENSOR", tool_id + ".csv"), encoding = "utf-8-sig" ) as f:
            reader = csv.reader(f)
            #next(reader)
            for row in reader:
                important_sensor.append(row[0])
        attribute = maximum_subset(deepcopy(attribute), deepcopy(important_sensor))
    else:
        print("can't not find \"" + str(tool_id) + "\"'s file of important_sensor!")
        
    print("@@@@@@@@@@@@ choose_important_sensor end! @@@@@@@@@@@@")
    
    return attribute



def save_data(data_folder_path, data_index, attribute, time_index):
    
    print("@@@@@@@@@@@@ save_data start! @@@@@@@@@@@@")
    
    
    ### prepare output directory
    
    if platform.system() == "Windows":
        save_data_path = r"\\10.100.10.66\sd1nas\TEMP_Joan\TRAINING_DATA\DATA\Blacksnail\input"
    else:
        #save_data_path = r"DATA/TEMP_Joan/TRAINING_DATA/DATA/Blacksnail/input"
        save_data_path = r"data/input"
    save_data_path = os.path.join(save_data_path, tool_id)
    
    if not os.path.exists(save_data_path):
        try:
            os.makedirs(save_data_path)
        except OSError as exc:  # Guard against race condition of path already existing
            if exc.errno != errno.EEXIST:
                raise
    
    
    ### reassign the attribute for all PDS (concatenate all PDS at the same time)
    ### plot data as well
    
    PDS_path = os.path.join(data_folder_path, "PDS")
    
    data = pd.DataFrame()
    for lot_idx, lot in enumerate(data_index):
        print("(" + str(lot_idx + 1) + "/" + str(len(data_index)) + ")")
        
        df = pd.read_csv(open(os.path.join(PDS_path, lot["PDS"])))
        
        ### reassign the attribute
        df = df[attribute]
        
        ### reassign the time
        df = df.iloc[time_index[lot["LOT_ID"]]["start"] : time_index[lot["LOT_ID"]]["end"]+1]
        
        ### plot data
        #plt.figure(figsize=(25, 8))
        #plt.plot(df)
        #plt.savefig( os.path.join(save_data_path, lot["LOT_ID"] + ".jpg") )
        #plt.close("all")
        
        ### concatenate data
        data = pd.concat([data, df])
        
        if lot_idx == 0:
            data_index[lot_idx]["start_index"] = 0
        else:
            data_index[lot_idx]["start_index"] = data_index[lot_idx - 1]["end_index"] + 1
        data_index[lot_idx]["end_index"] = data_index[lot_idx]["start_index"] + len(df) - 1
        data_index[lot_idx]["length"] = len(df)
    
    
    ### save data to csv file
    
    data.to_csv(path_or_buf = os.path.join(save_data_path, tool_id + "_original" + ".csv"), index = False)
    
    
    ### save data_index + attribute to json file
    
    for element in data_index:
        element["time"] = str(element["time"])
    
    with open(os.path.join(save_data_path, tool_id + "_original" + ".json"), "w") as output_file:
        json.dump(obj = {"data_index": data_index, \
                         "attribute": attribute, \
                         "time_index": time_index}, \
                  fp = output_file, \
                  indent = 4)
    
    
    print("@@@@@@@@@@@@ save_data end!   @@@@@@@@@@@@")
    
    

def data_preprocessing(data_folder_path, tool_id, start_time = None, end_time = None, input_mode = "ALL", output_mode = "save_data"):
    
    print("##########################################################################################################")
    print("##########################################################################################################")
    print("@@@@@@@@@@@@ data_preprocessing start! @@@@@@@@@@@@")
    
    
    ### load_data_index
    
    data_index = load_data_index(data_folder_path, tool_id, start_time, end_time, input_mode)
    
    
    ### integrate_attribute
    
    attribute = integrate_attribute(data_folder_path, tool_id, data_index)
    
    
    ### choose_important_time
    
    time_index = choose_important_time(data_folder_path, tool_id, data_index)
    
    
    ### choose_important_sensor
    
    attribute = choose_important_sensor(data_folder_path, tool_id, data_index, attribute)
    
    
    if output_mode == "save_data":
    
        ### plot_and_save_data
        
        save_data(data_folder_path, data_index, attribute, time_index)
        
    elif output_mode == "return_attribute":
    
        ### return_attribute
        
        return attribute
    
    
    print("@@@@@@@@@@@@ data_preprocessing end!   @@@@@@@@@@@@")
    print("##########################################################################################################")
    print("##########################################################################################################")



if __name__ == "__main__":
    
    if platform.system() == "Windows":
        nas_path = r"\\10.100.10.66\sd1nas"
    else:
        nas_path = r"DATA"
    
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", "DATA")
    
    tool_id = "GF"
    #start_time = "2017-12-10 00:00:00"
    #end_time = "2017-12-20 23:59:59"
    start_time = None
    end_time = None
    ### input_mode = "ALL" or "PDS"
    input_mode = "PDS"
    ### output_mode = "save_data" or "return_attribute"
    output_mode = "save_data"
    
    data_preprocessing(data_folder_path = data_folder_path, \
                       tool_id = tool_id, \
                       start_time = start_time, \
                       end_time = end_time, \
                       input_mode = input_mode, \
                       output_mode = output_mode)