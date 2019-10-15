import pandas as pd
# import brew_distance
import difflib
import re
from dateutil.parser import parse
# make it available to append a value to a key in a dict
from collections import defaultdict
from datetime import timedelta
import time

# there are log message, log key, value_vector, parameter_value_vectors in the log dataframe
# 1. how to generate log message: time(seconds)---- first one should be computed especially + message part
# 2. log key ---- read the corresponding line from structured csv
# 3. parameter_value_vectors ---- decrease log key part with original message part


# =========================== part to compute the log message ================================
def log_messages_create(fd):
    log_messages = []
    for month, date, time, message in zip(fd['Month'], fd['Date'], fd['Time'], fd['Content']):
        # process the () in message, in order to build matrix finally
        try:
            message = re.sub('(\(\))', '(0)', message)
        except Exception as e:
            print("there is an error: {}".format(e))
            pass
        log_messages.append(str(month)+' '+str(date)+' '+str(time)+' '+ str(message))
    return log_messages


# =========================== part to compute the parameter value vectors =========================
# function to compute the time gap
def time_gap(time_lines):
    # input is the series of time lines
    # output is the difference between two sequent times
    times = []
    # calculate the gap between two time
    for j in range(len(time_lines) - 1):
        times.append((time_lines[j + 1] - time_lines[j]))
    times = list(times)
    # we set the first value of time gap as 0
    times.insert(0,0)

    return times

# define the function to transfer the month name to month number
def month_string_to_number(string):
    m = {
        'Jan': 1,
        'Feb': 2,
        'Mar': 3,
        'Apr': 4,
        'May': 5,
        'Jun': 6,
        'Jul': 7,
        'Aug': 8,
        'Sep': 9,
        'Oct': 10,
        'Nov': 11,
        'Dec': 12
    }
    s = string.strip()[:3]

    try:
        out = m[s]
        return out
    except:
        pattern = '<.*>(.*)'
        match = re.match(pattern, string)
        s = match.group(1)
        out = m[s]
        return out

# function to convert the month, day, time into seconds
def trans_seconds(month_list, day_list, time_list):
    seconds_list = []
    seconds = 0
    for i in range(len(day_list)):
        # we assume there are 30 days for every month
        seconds = (month_list[i] - month_list[0]) * 30 * 24 * 3600 + (day_list[i] - day_list[0]) * 24 * 3600 + \
                  int(time_list[i][0]) * 3600 + int(time_list[i][1]) * 60 + int(time_list[i][2])
        # print("the seconds are:", seconds)
        seconds_list.append(seconds)
    return seconds_list


# function to get the variables (for parameter_value_vectors)
def show_diff(strings, strings1):
    # pattern = '[\ < \ * \ >, \(\ <.\ > \),:]'
    pattern = '[\<\*\>]'

    batch_variables = []
    for string, string1 in zip(strings, strings1):

        print("we are comparing two string %s and %s"%(string, string1))
        # we should delete the <*> |:| (<*>) part in a log key

        string1 = re.sub(pattern,'', string1)
        # string - string1
        seqm = difflib.SequenceMatcher(None, string, string1)
        output = []
        # define the difference list
        variables =[]
        for opcode, a0, a1, b0, b1 in seqm.get_opcodes():
            if a0 == a1:
                pass
            if opcode == 'equal':
                print("the sections [%d:%d] of string and [%d:%d] of string1 are the same" % (a0, a1, b0, b1))
            elif opcode == 'delete':
                print("remove %s from positions [%d:%d]" % (string[a0:a1], a0, a1))
                output.append(seqm.a[a0:a1])

        batch_variables.append(output)
        print("the batch_varibles are:",batch_variables)
    return batch_variables



# define the parameter value vector creation function
# we need use the variables in function single_match and append it after the time gap

# normally, we just analyse the system logs within one day, if we want to analyse logs beyond one day,
# we can extend the time process part
def parameter_vector(log_messages, log_keys, times, batch_variables, filename):
    '''
    :param log_messages: list of t + message
    :param log_keys: list of keys of raw logs
    :param times: list of time differences
    :param batch_variables: variables for the raw log and raw key
    :return: a dataframe with column names with raw_logs, log_keys, parameter_value_vectors
    '''

    parameter_value_vector = []
    parameter_value_vectors = []

    for time_v, variable in zip(times, batch_variables):
        print('we are appending time:',time_v)
        # time.sleep(5)
        print("we are appending variable:",variable)
        # time.sleep(5)
        parameter_value_vector = variable
        parameter_value_vector.insert(0,time_v)

        # print("the parameter_value_vector is:",parameter_value_vector)
        parameter_value_vectors.append(parameter_value_vector)

    # define the entry dict
    entry = {
        'log message': [],
        'log key': [],
        'parameter value vector': []
    }

    # transfer the entry to defaultdict
    entry = defaultdict(list)

    print("the parameter_value_vectors are:",parameter_value_vectors)
    for raw_log, log_key, parameter_value_vector_v in zip(log_messages, log_keys, parameter_value_vectors):
        # assign values to every row
        # print("we are writing raw_log into log_message:", raw_log)
        entry['log message'].append(raw_log)
        # print("we are writing log_key into log_key", log_key)
        entry['log key'].append(log_key)
        # print("we are writing parameter_value_vector into dataframe:", parameter_value_vector_v)
        # time.sleep(5)
        entry['parameter value vector'].append(parameter_value_vector_v)

    fieldnames = ['log message', 'log key', 'parameter value vector']
    df = pd.DataFrame(entry, columns=fieldnames)
    print("the entry is:", entry)
    # for normal linux system logs
    # df.to_csv('../data/System_logs/log_value_vector.csv', index=False)
    # for malicious linux system logs
    df.to_csv(filename, index=False)
    return df

if __name__ == '__main__':
    # load the structured_log csv
    # structured_log_filename = 'Dataset/Linux/Malicious_Separate_Structured_Logs/Integrated_structured_log.csv'
    print("the example is like: 'Dataset/xxx/Integrated_structured_log.csv', it is ’‘’")
    structured_log_filename = input("Please input the structured csv log: ")
    fd_linux = pd.read_csv(structured_log_filename)

    # create the first column in dataframe with 'log message'
    log_messages = log_messages_create(fd_linux)
    log_keys = fd_linux['EventTemplate']

    # append the timestamps to the log message
    month_list, day_list, time_list = [], [], []
    for i in range(len(fd_linux['Time'])):
        time_list.append(fd_linux['Time'][i].split(':'))
    for j in range(len(fd_linux['Date'])):
        day_list.append(fd_linux['Date'][j])
    # initialize the month_number
    month_number = 0

    # generate the first value(time_gap) in parameter log vector
    for k in range(len(fd_linux['Month'])):
        # print("we are transferring the month:",fd_linux['Month'][k])
        month_number = month_string_to_number(fd_linux['Month'][k])
        month_list.append(month_number)
    # call the generated month, day, time list to get the seconds
    seconds_list = trans_seconds(month_list, day_list, time_list)
    # input the seconds_list into the time_gap functions to get the series of time difference
    times = time_gap(seconds_list)
    # print("the times are:",times)
    # time.sleep(5)

    # process special case in log message to generate right values
    for i in range(len(fd_linux['Content'])):
        fd_linux['Content'][i] = re.sub('(\(\))', '(0)', fd_linux['Content'][i])
    fd_linux.to_csv(structured_log_filename, index=False)

    strings, strings1 = fd_linux['Content'], fd_linux['EventTemplate']
    # generate the later values in parameter value vector
    batch_variables = show_diff(strings=strings, strings1=strings1)

    print("the batch_variables are:", batch_variables)
    # time.sleep(5)

    # define the pattern of system log, we can change it depends on the format of system log
    system_log_pattern = '(\S+)(\s+)(\w+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(.*)'
    # generate the final dataframe for vector anomaly detection
    log_value_vector_filename = 'Dataset/Linux/Malicious_Separate_Structured_Logs/log_value_vector_mali.csv'
    df = parameter_vector(log_messages, log_keys, times, batch_variables, log_value_vector_filename)


