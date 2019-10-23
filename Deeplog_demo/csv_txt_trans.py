'''
This is the module to transform the client data to suitable format for whole anomaly detection
'''
import pandas as pd
import re
import sys
import os

sys.path.append('./')

def csv_to_txt(inputfile, outputfile):
    # get the message parts from inputfile
    message_list, timestamp_list = list(), list()
    month, date, time = None, None, None
    print("we are prcoessing {}".format(inputfile))
    fd = pd.read_csv(inputfile)
    message_list = fd['message']
    timestamp_list = fd['@timestamp']
    # match the message part 5
    pattern = "(\<.*\>)(\w+)(\s+)(\d+) (\d+:\d+:\d+) (.*)"
    # match the timestamp 1,2,4
    pattern1 = '(\w+) (\d+),(.*) (\d+:\d+:\d+)'
    with open(outputfile, 'w') as output_file:
        for message, timestamp in zip(message_list, timestamp_list):
            # print("the timestamp is:", timestamp)
            message = re.match(pattern, message).group(6)
            month = re.match(pattern1, timestamp).group(1)
            date = re.match(pattern1, timestamp).group(2)
            time = re.match(pattern1, timestamp).group(4)
            timestamp = month+' '+ date +' '+time
            output_file.write(timestamp + ' ' + message+'\n')


def batch_processing(filename, batch_dir):
    lines = []
    with open(filename, 'r+') as f:
        lines = f.readlines()

    length = len(lines)
    print("the length of txt is:", length)
    # define the batch number
    i = 0
    while i * 2000 <= length:
        with open(batch_dir+'batch'+'_'+ str(i) +'.txt', 'w') as fw:
            for row in lines[i * 2000 : (i+1) * 2000]:
                fw.write(row)
        i += 1


if __name__ == '__main__':
    # input_dir = input("Please input the path of the file you desire to process:")
    # out_dir = input("Please input the path of filename you desire to output:")
    print("the example of input_dir is like: Dataset/Linux/Client/Client_data/")
    input_dir = input('Please input the path where the client_data lies in: ')
    print("the example of output_dir is like: Dataset/Linux/Client/Client_transformed/")
    output_dir = input('Please input the path where the transformed client data lies in: ')
    print("the example of batch_dir is like: Dataset/Linux/Client/Client_batch_txt/")
    batch_dir = input("Please input the path where the batched data lies in: ")
    # loop to transform csv to txt files
    for r, d, f in os.walk(input_dir):
        for file in f:
            if file.endswith('.csv'):
                # splitext will delete the suffix csv
                inputfile = input_dir + file
                outputfile = output_dir + os.path.splitext(file)[0] +'.txt'
                csv_to_txt(inputfile, outputfile)
    # loop to split txt into fixed batch sized txt files
    for r, d, f in os.walk(output_dir):
        for file in f:
            if file.endswith('.txt'):
                filename = output_dir + file
                print("we are processing:", filename)
                batch_processing(filename, batch_dir + os.path.splitext(file)[0]+'_')

