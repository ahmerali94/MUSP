import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd

# ProvinoD file
filename = "C:/MUSP_Local/Data/Ahmar/Data/ProvinoD.h5"

with h5py.File(filename, "r") as f:
    # list of all the keys in the H5 file : only one group in this file name PovinoD
    print("Keys: %s" % f.keys())
    data_group_name = list(f.keys())[0]
    display('Names of the main group: \n ', data_group_name)

    # list all the keys in the group 'provinoD' e.g AccelerationX, AccelerationY.....
    data_sub_groups = list(f[data_group_name])
    display('Names of the keys in the main group: \n ', data_sub_groups)

    # list all the items in the group 'ProvinoD'
    data_base_items = list(f.items())
    display('Items in the group: \n', data_base_items)

    # get data in the main group
    data_main = f.get(data_group_name)

    # all the data in the main group
    data_items = list(data_main.items())
    display('Data in the group: \n', data_items)

    print('Total number of keys in the group \n', len(data_sub_groups))

    # making numpy array of all the keys' values

    i = 0
    data = [None] * len(data_sub_groups)

    while i < (len(data_sub_groups)):
        data[i] = np.array(data_main.get(data_sub_groups[i]))
        i += 1

    # change list elements to pandas dataframe and transpose the rows and cols
    data_df = pd.DataFrame(data)
    data_df = data_df.T

    # cols name from the H5 file
    data_df.columns = data_sub_groups
    data_df.to_csv('C:/MUSP_Local/Data/CSV_Files/ProvinoD.csv', index=False)

# ProvinoF file
filename = "C:/MUSP_Local/Data/Ahmar/Data/ProvinoF.h5"

with h5py.File(filename, "r") as f:
    # list of all the keys in the H5 file : only one group in this file name PovinoD
    print("Keys: %s" % f.keys())
    data_group_name = list(f.keys())[0]
    display('Names of the main group: \n ', data_group_name)

    # list all the keys in the group 'provinoD' e.g AccelerationX, AccelerationY.....
    data_sub_groups = list(f[data_group_name])
    display('Names of the keys in the main group: \n ', data_sub_groups)

    # list all the items in the group 'ProvinoD'
    data_base_items = list(f.items())
    display('Items in the group: \n', data_base_items)

    # get data in the main group
    data_main = f.get(data_group_name)

    # all the data in the main group
    data_items = list(data_main.items())
    display('Data in the group: \n', data_items)

    print('Total number of keys in the group \n', len(data_sub_groups))

    # making numpy array of all the keys' values

    i = 0
    data = [None] * len(data_sub_groups)

    while i < (len(data_sub_groups)):
        data[i] = np.array(data_main.get(data_sub_groups[i]))
        i += 1

    # change list elements to pandas dataframe and transpose the rows and cols
    data_df = pd.DataFrame(data)
    data_df = data_df.T

    # cols name from the H5 file
    data_df.columns = data_sub_groups
    data_df.to_csv('C:/MUSP_Local/Data/CSV_Files/ProvinoF.csv', index=False)

# ProvinoH file
filename = "C:/MUSP_Local/Data/Ahmar/Data/ProvinoH.h5"

with h5py.File(filename, "r") as f:
    # list of all the keys in the H5 file : only one group in this file name PovinoD
    print("Keys: %s" % f.keys())
    data_group_name = list(f.keys())[0]
    display('Names of the main group: \n ', data_group_name)

    # list all the keys in the group 'provinoD' e.g AccelerationX, AccelerationY.....
    data_sub_groups = list(f[data_group_name])
    display('Names of the keys in the main group: \n ', data_sub_groups)

    # list all the items in the group 'ProvinoD'
    data_base_items = list(f.items())
    display('Items in the group: \n', data_base_items)

    # get data in the main group
    data_main = f.get(data_group_name)

    # all the data in the main group
    data_items = list(data_main.items())
    display('Data in the group: \n', data_items)

    print('Total number of keys in the group \n', len(data_sub_groups))

    # making numpy array of all the keys' values

    i = 0
    data = [None] * len(data_sub_groups)

    while i < (len(data_sub_groups)):
        data[i] = np.array(data_main.get(data_sub_groups[i]))
        i += 1

    # change list elements to pandas dataframe and transpose the rows and cols
    data_df = pd.DataFrame(data)
    data_df = data_df.T

    # cols name from the H5 file
    data_df.columns = data_sub_groups
    data_df.to_csv('C:/MUSP_Local/Data/CSV_Files/ProvinoH.csv', index=False)

# ProvinoH_G file
filename = "C:/MUSP_Local/Data/Ahmar/Data/ProvinoH_G.h5"

with h5py.File(filename, "r") as f:
    # list of all the keys in the H5 file : only one group in this file name PovinoD
    print("Keys: %s" % f.keys())
    data_group_name = list(f.keys())[0]
    display('Names of the main group: \n ', data_group_name)

    # list all the keys in the group 'provinoD' e.g AccelerationX, AccelerationY.....
    data_sub_groups = list(f[data_group_name])
    display('Names of the keys in the main group: \n ', data_sub_groups)

    # list all the items in the group 'ProvinoD'
    data_base_items = list(f.items())
    display('Items in the group: \n', data_base_items)

    # get data in the main group
    data_main = f.get(data_group_name)

    # all the data in the main group
    data_items = list(data_main.items())
    display('Data in the group: \n', data_items)

    print('Total number of keys in the group \n', len(data_sub_groups))

    # making numpy array of all the keys' values

    i = 0
    data = [None] * len(data_sub_groups)

    while i < (len(data_sub_groups)):
        data[i] = np.array(data_main.get(data_sub_groups[i]))
        i += 1

    # change list elements to pandas dataframe and transpose the rows and cols
    data_df = pd.DataFrame(data)
    data_df = data_df.T

    # cols name from the H5 file
    data_df.columns = data_sub_groups
    data_df.to_csv('C:/MUSP_Local/Data/CSV_Files/ProvinoH_G.csv', index=False)

# ProvinoN file
filename = "C:/MUSP_Local/Data/Ahmar/Data/ProvinoN.h5"

with h5py.File(filename, "r") as f:
    # list of all the keys in the H5 file : only one group in this file name PovinoD
    print("Keys: %s" % f.keys())
    data_group_name = list(f.keys())[0]
    display('Names of the main group: \n ', data_group_name)

    # list all the keys in the group 'provinoD' e.g AccelerationX, AccelerationY.....
    data_sub_groups = list(f[data_group_name])
    display('Names of the keys in the main group: \n ', data_sub_groups)

    # list all the items in the group 'ProvinoD'
    data_base_items = list(f.items())
    display('Items in the group: \n', data_base_items)

    # get data in the main group
    data_main = f.get(data_group_name)

    # all the data in the main group
    data_items = list(data_main.items())
    display('Data in the group: \n', data_items)

    print('Total number of keys in the group \n', len(data_sub_groups))

    # making numpy array of all the keys' values

    i = 0
    data = [None] * len(data_sub_groups)

    while i < (len(data_sub_groups)):
        data[i] = np.array(data_main.get(data_sub_groups[i]))
        i += 1

    # change list elements to pandas dataframe and transpose the rows and cols
    data_df = pd.DataFrame(data)
    data_df = data_df.T

    # cols name from the H5 file
    data_df.columns = data_sub_groups
    data_df.to_csv('C:/MUSP_Local/Data/CSV_Files/ProvinoN.csv', index=False)

