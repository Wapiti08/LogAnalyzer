#!/usr/bin/env python
import sys
sys.path.append('../')
from logparser import LenMa
import os
import joblib
import re
import pandas as pd

# input_dir  = '../logs/HDFS/' # The input directory of log file
# output_dir = 'Lenma_result/' # The output directory of parsing results
# log_file   = 'HDFS_2k.log' # The input log file name
# log_format = '<Date> <Time> <Pid> <Level> <Component>: <Content>' # HDFS log format
# threshold  = 0.9 # TODO description (default: 0.9)
# regex      = [] # Regular expression list for optional preprocessing (default: [])


# input_dir  = '../logs/Linux/'
# output_dir = 'Lenma_result/'
# # log_file   = 'malicious_linux.log'
# # log_file   = 'Linux_100k_part.log'
# # log_file = 'malicious_linux_test.log'
# log_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
# regex      = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
# threshold  = 0.88

# generate the structured log csv for malicious logs
# input_dir = '../../Dataset/Linux/Malicious/'
# output_dir = '../../Dataset/Linux/Malicious_Separate_Structured_Logs/'


# # generate the structured log csv for benign logs
input_dir = '../../Dataset/Linux/Clear/'
# make sure under the directory, the filename is like xx1.xx, xx2.xx (include digit numbers to integrate then)
output_dir = '../../Dataset/Linux/Clear_Separate_Structured_Logs/'



# r == root, d == directories, f = files
suspicious_positions = []
# record the file with obvious malicious logs in
file_num = []
for r, d, f in os.walk(input_dir):
    if os.path.isfile(output_dir+'suspicious_positions.pkl'):
        suspicious_positions = joblib.load(output_dir+'suspicious_positions.pkl')
        print("the suspicious file names include (only number) {}".format(suspicious_positions))
    else:
        for file in f:
            log_file = file
            log_format = '<Month> <Date> <Time> <Level> <Component>(\[<PID>\])?: <Content>'
            regex = [r'(\d+\.){3}\d+', r'\d{2}:\d{2}:\d{2}']
            threshold = 0.88
            try:
                parser = LenMa.LogParser(input_dir, output_dir, log_format, threshold=threshold, rex=regex)
                parser.parse(log_file)
            except Exception as e:
                print("there is a suspicious malicious log, we got the error like: {}, please check the corresponding position".format(e))
                file_num = int(re.search('\d+', log_file).group(0))
                suspicious_positions.append(file_num)
                # suspicious_positions.append(e)
                continue
        suspicious_positions = sorted(suspicious_positions)
        print("the suspicious file names include (only number) {}".format(suspicious_positions))
        joblib.dump(suspicious_positions, '../../Dataset/Linux/Malicious_Separate_Structured_Logs/suspicious_positions.pkl')

#  the module to integrate all single-structured log files into one file according to timestamps
# the order is really depending on the practical situation, here it is decreasing order


# integration part:
structured_csv_list = []
columns = ['LineId', 'Month', 'Date', 'Time', 'Level', 'Component', 'PID', 'Content', 'EventId', 'EventTemplate']
csv_out = 'Integrated_structured_log.csv'

# got the structured log files list
for r, d, f in os.walk(output_dir):
    for file in f:
        if file.endswith('_structured.csv'):
            structured_csv_list.append(file)

# print("the structured_csv_list is:", structured_csv_list)
# create the dict with index(extracted number from filename): filename
ind_file_dict = {}
for filename in structured_csv_list:

    if re.search(r'\d+', filename):
        ind_file_dict[re.search(r'\d+', filename).group(0)] = filename
    # exception to process the filename without digit, normally it is the current log set
    else:
        ind_file_dict['0'] = filename

# print("the ind_file_dict is:",ind_file_dict)

int_csv = []
# integrate the csv file as the index decreases
# transform the str element into int type, which will be used to sort
ind_file_dict_keys = [int(key) for key in ind_file_dict.keys()]
index_list = sorted(ind_file_dict_keys, reverse = True)
# print("the indexed list is:", index_list)
for i in range(len(index_list)):
    index = str(index_list[i])
    # print("we are integrating csv file:",ind_file_dict[index])
    df = pd.read_csv(output_dir+ind_file_dict[index], index_col = None, header=0)
    int_csv.append(df)

frame = pd.concat(int_csv, axis=0, ignore_index=True)
# delete the index column
frame.to_csv(output_dir+csv_out, header = columns, index =False )
