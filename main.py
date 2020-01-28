import importlib
import platform
import os


if __name__ == "__main__":
    
    # load data
    
    load_data = importlib.import_module("package.3-load_data")
    
    if platform.system() == "Windows":
        nas_path = r"\\10.100.10.66\sd1nas"
    else:
        nas_path = r"DATA"
    data_folder_path = os.path.join(nas_path, "TEMP_Joan", "TRAINING_DATA", "DATA", "Blacksnail")
    
    tool_id = "test"
    
    data = load_data.load_data(data_folder_path = data_folder_path, \
                               tool_id = tool_id)
    
    print(data[0][0])
    
    '''
    # RCAE2E
    
    RCAE2E = importlib.import_module("package.4-RCAE2E")
    
    
    r_w = 5
    t_w = 10
    
    k = 5
    _lambda = 0.1
    beta = 1
    alpha = 1
    
    c = 0.9
    
    RCAE2E.RCAE2E(data = data, \
                  r_w = r_w, \
                  t_w = t_w, \
                  k = k, \
                  _lambda = _lambda, \
                  beta = beta, \
                  alpha = alpha, \
                  c = c)
    '''