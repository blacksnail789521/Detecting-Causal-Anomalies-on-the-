import sys
import pandas as pd
from shutil import copyfile
import json
from pathlib import Path
import pickle
import platform
import os


def get_PDS_data(nas_path, data_folder_name, number_of_wafer):
    
    print("@@@@@@@@@@@@ get_PDS_data start! @@@@@@@@@@@@")
    
    
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", data_folder_name)
    
    ### check folder
    
    if not os.path.exists(os.path.join(data_folder_path, "PDS")):
        os.makedirs(os.path.join(data_folder_path, "PDS"))
    
    
    ### get uniqueLots
    
    frames = []
    
    candidate_list_path = os.path.join(data_folder_path, "candidate_list")
    
    for dirpath, dirnames, files in os.walk(candidate_list_path):
        for file in files:
            df = pd.read_csv(open(os.path.join(candidate_list_path, file)))
            frames.append(df["LOT_ID"])
    
    uniqueLots = list(pd.concat(frames).unique())
    
    
    ### get PDS
    
    EQP_ID = []
    RUN_ID = []
    LOT_ID = []
    
    for uniqueLot in uniqueLots:
        EQP_ID.append(uniqueLot[0:2])
        RUN_ID.append(uniqueLot[0:2] + "_" + uniqueLot)
        LOT_ID.append(uniqueLot)
    
    if number_of_wafer == None:
        length_of_wafer = len(EQP_ID)
    else:
        length_of_wafer = number_of_wafer if number_of_wafer < len(EQP_ID) else len(EQP_ID)
    
    PDS_index_path = os.path.join(data_folder_path, "PDS", "PDS_index.json")
    
    PDS_index = {"valid": [], "invalid": []}
    
    ### past_PDS_index is used for checking the file is done or not
    if os.path.isfile(PDS_index_path):
        with open(PDS_index_path, "r") as input_file:
            past_PDS_index = json.load(input_file)
    else:
        past_PDS_index = None
    
    for i in range(length_of_wafer):
        
        ### check past_PDS_index
        if past_PDS_index != None:
            past_PDS_index_only_lot_id = [element["lot_id"] for element in past_PDS_index["valid"]]
            if LOT_ID[i] in past_PDS_index_only_lot_id:
                print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + \
                      RUN_ID[i] + ": processed already! (valid)")
                PDS_index["valid"].append(past_PDS_index["valid"][past_PDS_index_only_lot_id.index(LOT_ID[i])])
                continue
            elif LOT_ID[i] in past_PDS_index["invalid"]:
                print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + \
                      RUN_ID[i] + ": processed already! (invalid)")
                PDS_index["invalid"].append(LOT_ID[i])
                continue
            else:
                pass
        
        PDS_path = os.path.join(nas_path, "PDS_ARCH", EQP_ID[i])
        
        find_the_file = False
        
        for file_name in os.listdir(PDS_path):
        
            #print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": " )
            
            if file_name.find(RUN_ID[i]) != -1:
                
                find_the_file = True
                
                ### check whether the file is empty or not
                data_size = os.path.getsize(os.path.join(PDS_path, file_name))
                if data_size == 0:
                    print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": data size = 0")
                    PDS_index["invalid"].append(LOT_ID[i])
                    continue
                
                
                data_length = pd.read_csv(open(os.path.join(PDS_path, file_name))).shape[0] - 1
                
                ### check data_length
                if data_length < 10:
                    print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": data length < 10")
                    PDS_index["invalid"].append(LOT_ID[i])
                else:
                    ### data_length is legal, so now we check miss_rate and sample_rate
                    ### determine miss_rate and sample_rate
                    if file_name.find("#Y") != -1:
                        miss_rate = int(file_name[file_name.find("#Y") + 2 : file_name.find(".csv")]) / data_length
                        if miss_rate > 1:
                            miss_rate = 1
                        sample_rate = int(file_name[file_name.find("#") + 1 : file_name.find("#Y")])
                    else:
                        miss_rate = 0
                        sample_rate = int(file_name[file_name.find("#") + 1 : file_name.find(".csv")])
                    
                    ### check miss_rate
                    if miss_rate > 0.3:
                        print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": (miss rate = " + "%.2f" % (miss_rate * 100) + "%, " + \
                              "sample rate = " + "%d" % sample_rate + ") too many missing data (>30%)")
                        PDS_index["invalid"].append(LOT_ID[i])
                    
                    ### check sample_rate
                    elif sample_rate != 1:
                        print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": (miss rate = " + "%.2f" % (miss_rate * 100) + "%, " + \
                              "sample rate = " + "%d" % sample_rate + ") sample rate != 1")
                        PDS_index["invalid"].append(LOT_ID[i])
                    
                    ### both miss_rate and sample_rate are valid!
                    else:
                        
                        print("(" + str(i + 1) + "/" + str(length_of_wafer) + ") " + RUN_ID[i] + ": (miss rate = " + "%.2f" % (miss_rate * 100) + "%, " + \
                              "sample rate = " + "%d" % sample_rate + ") legal!")
                        
                        ### data_preprocessing
                        
                        df = pd.read_csv(open(os.path.join(PDS_path, file_name)))
                        """
                        colume_is_time_series = []
        
                        stop = False
                        for index, row in df.iterrows():
                            for attribute in list(df):
                                if str(df[attribute].dtype) != "float64" and row[attribute] == "@@":
                                    stop = True
                                    colume_is_time_series.append(attribute)
                            
                            if stop == True:
                                break
                        
                        for attribute in colume_is_time_series:
                            df[attribute] = pd.to_numeric(df[attribute], errors='coerce').interpolate()
                        """
                        
                        for attribute in list(df):
                            if attribute not in ["EQP_ID", "START_TIME", "END_TIME", "LOT_ID", "RECIPE", \
                                                 "RUN_ID", "STEP_NUM", "LAYER", "TRACE_DATE", "TRACE_SEC"]:
                                df[attribute] = pd.to_numeric(df[attribute], errors='coerce').interpolate()
                        
                        
                        df.to_csv(path_or_buf = os.path.join(data_folder_path, "PDS", file_name), index = False)
                        
                        PDS_index["valid"].append({"time": df["TRACE_DATE"].iloc[0], "lot_id": LOT_ID[i]})
                    
        if find_the_file == False:
            print("not exist!")
            PDS_index["invalid"].append(LOT_ID[i])
    
        with open(PDS_index_path, "w") as output_file:
            json.dump(obj = PDS_index, fp = output_file, indent = 4)

    
    print("@@@@@@@@@@@@ get_PDS_data end!   @@@@@@@@@@@@")



def get_PL_data(nas_path, data_folder_name, number_of_wafer):
    
    print("@@@@@@@@@@@@ get_PL_data start! @@@@@@@@@@@@")
    
    
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", data_folder_name)
    
    def checked(cl, wid):
        for run in cl:
            if run["lot_id"] in wid:
                return True
        return False
    
    
    # check folder
    
    if not os.path.exists(os.path.join(data_folder_path, "PL")):
        os.makedirs(os.path.join(data_folder_path, "PL"))
    
    
    ### get plInfo
    
    frames = []
    
    candidate_list_path = os.path.join(data_folder_path, "candidate_list")
    
    for dirpath, dirnames, files in os.walk(candidate_list_path):
        for file in files:
            df = pd.read_csv(open(os.path.join(candidate_list_path, file)))
            frames.append(df[["WFR_ID", "PLFilePath"]])
    
    plInfo = pd.concat(frames).drop_duplicates()
    
    
    ### get PL
    
    PDS_index_path = os.path.join(data_folder_path, "PDS", "PDS_index.json")
    
    with open(PDS_index_path) as file:
        checkList = json.load(file)['valid']
    
    src = os.path.join(nas_path, "DIE_LEVEL", "PL_DATA", "H")
    dst = os.path.join(data_folder_path, "PL")
    
    if number_of_wafer == None:
        lenPl = len(plInfo)
    else:
        lenPl = number_of_wafer if number_of_wafer < len(plInfo) else len(plInfo)
    
    hasPl = []
    
    for i in range(lenPl):
        temp = plInfo.iloc[i]
        if checked(checkList, temp['WFR_ID']):
            
            S = os.path.join(src, str(temp['PLFilePath']), temp['WFR_ID'] + '.txt')
            D = os.path.join(dst, temp['WFR_ID'] + '.txt')
            
            f = Path(S)
            if f.exists():
                copyfile(S, D)
                print('Processing...' + temp['WFR_ID'])
                hasPl.append(temp['WFR_ID'])
            else:
                print(temp['WFR_ID'] + '.txt not exists...')
        else:
            print(temp['WFR_ID'] + ' does not have PDS data...')
    
    print('Write PL_index file...')
    
    PL_index_path = os.path.join(data_folder_path, "PL", "PL_index.json")
    
    with open(PL_index_path, 'w') as out:
        json.dump(obj = hasPl, fp = out, indent = 4)

    print("@@@@@@@@@@@@ get_PL_data end!   @@@@@@@@@@@@")


    
def merge_data(nas_path, data_folder_name):
    
    print("@@@@@@@@@@@@ merge_data start! @@@@@@@@@@@@")
    
    
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", data_folder_name)
    
    def retFullName(pdsFiles, lid):
        l = len(pdsFiles)
        for i in range(l):
            if lid in pdsFiles[i]:
                    break
        return pdsFiles[i]
    
    
    def getTime(pdsIndex, lot):
        for dic in pdsIndex:
            if lot == dic['lot_id']:
                return dic['time']
    
    
    ### check folder
    
    if not os.path.exists(os.path.join(data_folder_path, "ALL")):
        os.makedirs(os.path.join(data_folder_path, "ALL"))
    
    
    
    PDS_index_path = os.path.join(data_folder_path, "PDS", "PDS_index.json")
    PL_index_path = os.path.join(data_folder_path, "PL", "PL_index.json")

    with open(PDS_index_path, 'r') as file:
        print("Read file... " + os.path.join(data_folder_path, "PDS", "PDS_index.json"))
        pds = json.load(file)["valid"]
        
    with open(PL_index_path, 'r') as file:
        print("Read file..." + os.path.join(data_folder_path, "PL", "PL_index.json"))
        pl = json.load(file)
    
    pdsFiles = os.listdir(os.path.join(data_folder_path, "PDS"))
    
          
    ### get tools and runs 
    ### format {tool: [run_ID, run_ID,...], tool: [...]}
    tools = {}
    lenPDS = len(pds)
    for i in range(lenPDS):
        t = pds[i]["lot_id"][: 2]
        if t in tools:
            tools[t].append(int(pds[i]["lot_id"][2:]))
        else:
            tools[t] = [int(pds[i]["lot_id"][2:])]
    
    for k, v in tools.items():
        tools[k].sort()
        
    ### transform pl list into dic with
    ### format {Lot_ID: [wafer_ID, wafer_ID,...], Lot_ID: [...], ...}
    ### wafer_ID belongs to {0100, ..., 1400}
    pldic = {}
    
    lenPl = len(pl)
    for i in range(lenPl):
        lid = pl[i][: -4]
        wid = pl[i][-4:]
        if lid in pldic:
            pldic[lid].append(wid)
        else:
            pldic[lid] = [wid]
    
    
    
    ### start packaging data
    
    frames = []
    
    MAP_path = os.path.join(data_folder_path, "MAP")
    
    for dirpath, dirnames, files in os.walk(MAP_path):
        for file in files:
            frames.append(pd.read_csv(open(os.path.join(MAP_path, file))))
    
    mapD = pd.concat(frames).drop_duplicates()
    
    
    
    
    
    ### store data in .json file, one tool one file.
    ### format [{time: , Lot_ID: , PDS: , waferinfo: [[wafer_ID, MAP data, PL data]]}]
    ### wafer_ID belongs to {0100, ..., 1400}
    ### MAP data: a pandas
    ### PL data: a pandas dataframe
    for tool, runs in tools.items():
        dataset = []
        print('Merging ' + tool + '...')
        for r in runs:
            rs = str(r)
            
            #tempPDS = pd.read_csv(open(os.path.join(data_folder_path, "PDS", retFullName(pdsFiles, tool + rs))))
            
            thisRun = mapD[mapD['WFR_ID'].str.contains(tool + rs)]
            winfo = []
            for idx, wafer in thisRun.iterrows():
                wid = wafer['WFR_ID'][-4:]
                tw = {'id': wid, 'MAP': wafer, 'PL': None}      
                if tool + rs in pldic and wid in pldic[tool + rs]:
                    img = pd.read_csv(open(os.path.join(data_folder_path, "PL",tool + rs + wid + ".txt") \
                                           , encoding = "utf-8"), header=25)
                    
                    img = img.drop(img.index[0]).drop('Unnamed: 8', axis=1)
                    tw['PL'] = img
                winfo.append(tw)
            dataset.append({'time': getTime(pds, tool + rs), 
                            'lot': str(r), 
                            'PDS': retFullName(pdsFiles, tool + rs), 
                            'WINFO': winfo})
            """logger.info('\tRun ' + str(r) + '...')"""
            print('\tRun ' + str(r) + '...')
        
        with open(os.path.join(data_folder_path, "ALL", tool + ".pickle"), 'wb') as file:
            print(os.path.join("Save file to ", data_folder_path, "ALL", tool, ".pickle"))
            pickle.dump(dataset, file)
    
    print("@@@@@@@@@@@@ merge_data end!   @@@@@@@@@@@@")



def get_data(nas_path, data_folder_name, number_of_wafer = None, mode = "ALL"):
    
    print("##########################################################################################################")
    print("##########################################################################################################")
    print("@@@@@@@@@@@@ get_data start! @@@@@@@@@@@@")
    
    
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", data_folder_name)
    
    ### check files' existence
    
    for dirpath, dirnames, files in os.walk(data_folder_path + "\\candidata_list"):
        if not files:
            print("candidate_list does not exist!")
            sys.exit()
    
    for dirpath, dirnames, files in os.walk(data_folder_path + "\\MAP"):
        if not files:
            print("MAP does not exist!")
            sys.exit()
    
    print('candidate_list and MAP exist!')
    
    
    ### different mode
    
    if mode == "ALL":
        
        get_PDS_data(nas_path, data_folder_name, number_of_wafer)
        get_PL_data(nas_path, data_folder_name, number_of_wafer)
        merge_data(nas_path, data_folder_name)
        
    elif mode == "PDS":
        
        get_PDS_data(nas_path, data_folder_name, number_of_wafer)
        
    elif mode == "PL":
        
        get_PL_data(nas_path, data_folder_name, number_of_wafer)
        
    elif mode == "MERGE":
        
        merge_data(nas_path, data_folder_name)
    
    else:
        print("mode error!")
    
    
    
    print("@@@@@@@@@@@@ get_data end!   @@@@@@@@@@@@")
    print("##########################################################################################################")
    print("##########################################################################################################")



if __name__ == "__main__":
    
    if platform.system() == "Windows":
        nas_path = r"\\10.100.10.66\sd1nas"
    else:
        nas_path = r"/home/mladmin/22289/DATA"
    data_folder_name = "DATA_NEW"
    
    number_of_wafer = 5
    mode = "ALL"
    
    get_data(nas_path = nas_path, \
             data_folder_name = data_folder_name, \
             number_of_wafer = number_of_wafer, \
             mode = mode)