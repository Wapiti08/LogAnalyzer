'''
This part is to generate the matrix for every log key and store them under
a directory
'''

import pandas as pd
import numpy as np
import re
from pandas import Series
import os


def log_vectors(fd_values, filename):
    '''
    :param fd_values: csv with numerical values in a single parameter value vector column
    :return: key_num_para_dict: the format is {Exx:[numerical parameter 1],[numerical parameter 2],...}
    '''
    uni_log_key_id = list(set(fd_values['log key']))

    parameters = []

    key_num_para_dict = {}
    for i in range(len(uni_log_key_id)):
        # get all the parameters with the same eventID
        parameters = fd_values[fd_values['log key'] == uni_log_key_id[i]]['parameter value vector']
        # the format will be E88: [xx,xx,xx],[xf,er,et],[re,tet,tet],...
        key_num_para_dict[uni_log_key_id[i]] = parameters.values[:]
    # plt.hist(key_para_dict['E88'], bins=15)
    # plt.ylable('Counter')
    # plt.xlabel('Variables')
    # plt.title('Single Log Key Vectors')
    print("the num para dict is:",key_num_para_dict)

    df_dict_num_para = pd.DataFrame(dict([(k, Series(v)) for k, v in key_num_para_dict.items()]))
    # df_dict_num_para.to_csv('../Dataset/Linux/Malicious_Separate_Structured_Logs/key_num_para_dict.csv', index= False, header=key_num_para_dict.keys())
    df_dict_num_para.to_csv(filename, index= False, header=key_num_para_dict.keys())
    return key_num_para_dict


# define the module to transform str into matrix
# the string is like: '10635,[21, 85, 16, 18],[21, 85, 16, 18, 307, 308, 1],[356],[424],[207]'
def str_array(dict, eventID, filename):
    '''
    :param dict: the format is key:[numerical parameter 1],[numerical parameter 2],..
    :param eventID: Exx
    :return: saved matrix for unique event
    '''
    lists = dict[eventID]
    list_string = []
    pattern = '\d+'
    numx = len(lists)
    numy = len(re.findall(pattern, lists[0]))
    list_array = np.empty(shape=[0, numy])
    for string in lists:
        # matching all digits
        list_string = re.findall(pattern, string)
        # transform the str into int
        list_string = [int(var) for var in list_string]
        list_string = np.array(list_string)
        # concatenate multi lines
        try:
            list_array = np.append(list_array, [list_string], axis = 0)
        except Exception as e:
            print("there is an error like:", e)
            pass
    # np.save('../Dataset/Linux/Malicious_Separate_Structured_Logs/Event_npy/'+eventID+'.npy', list_array)
    if os.path.exists(filename):
        np.save(filename + eventID+'.npy', list_array)
    else:
        os.mkdir(filename)
        np.save(filename + eventID+'.npy', list_array)


if __name__ == '__main__':
    # get all the parameter value vector for every unique key log
    filename = '../Dataset/Linux/Client/Client_structured/log_value_vector.csv'
    fd_values = pd.read_csv(filename)
    # create the aim file where the key_num_para_dict.csv will be saved
    para_dict_filename = input("Please input the folder to save the key_num_para_dict.csv: ")
    key_num_para_dict = log_vectors(fd_values, para_dict_filename)

    Event_npy_folder = input("Please input the folder to save the event matrix for every log key: ")
    # create all the matrixes for all the eventIDs
    for key in key_num_para_dict.keys():
        print("the key is:", key)
        str_array(key_num_para_dict, key, Event_npy_folder)


'''
Aug 9 7:25:03 authentication failure; logname= uid=0 euid=0 tty= ruser= rhost=  user=nobody
Aug 8 14:18:53 authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=63.251.144.88  user=root
'''
