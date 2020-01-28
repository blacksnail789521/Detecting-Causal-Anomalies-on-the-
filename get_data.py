import importlib
import platform


if __name__ == "__main__":
    
    # log2File
    
    log2File = importlib.import_module("package.0-log2File")
    log2File.log2File('get_data')
    
    
    # get_data
    
    if platform.system() == "Windows":
        nas_path = r"\\10.100.10.66\sd1nas"
    else:
        nas_path = r"/home/mladmin/22289/DATA"
    data_folder_name = "DATA"
    
    number_of_wafer = None
    mode = "PDS"
    
    get_data = importlib.import_module("package.1-get_data")
    
    get_data.get_data(nas_path = nas_path, \
                      data_folder_name = data_folder_name, \
                      number_of_wafer = number_of_wafer, \
                      mode = mode)