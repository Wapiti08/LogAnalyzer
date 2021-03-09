import pandas as pd
import brew_distance
import difflib
import re
from dateutil.parser import parse
# make it available to append a value to a key in a dict
from collections import defaultdict
from datetime import timedelta

# we define the distance within inner-class cluster is 12 according to the test we did
# while when we use 12 to implement program, which we found it is not so practical
# threshold = 12
# if we have generated all the log key for a log file
# when we try to match one(available always), we should use the minimum distance between distances


# define the function to find the difference between two logs
def show_diff(string, string1):
    print("we are comparing two string %s and %s"%(string, string1))
    seqm = difflib.SequenceMatcher(None, string ,string1)
    output = []
    # define the difference list
    variables =[]
    for opcode, a0,a1, b0,b1 in seqm.get_opcodes():
        print("the deleted part is:", seqm.a[a0:a1])
        print("the deleted position is %d and %d"%(a0,a1))
        if a0 == a1:
            pass
        elif opcode == 'equal':
            pass
        elif opcode == 'delete':
            output.append(seqm.a[a0:a1]+' ')
            pass
        else:
            #raise RuntimeError("unexpected opcode")
            pass
    variables = ''.join(output)
    variables = variables.split()
    return variables

# define the function to calculate the time gap

def time_gap(time_filename):
    times = []
    with open(time_filename) as timefile:
        lines = timefile.readlines()
    # tranform the strin into datetime format
    for i in range(len(lines)):
        lines[i] = parse(lines[i])
    print("the length of time lines are:", len(lines))
    # calculate the gap between two time
    for j in range(len(lines) - 1):
        times.append((lines[j + 1] - lines[j]).seconds)
    times = list(times)
    # we set the first value of time gap as 0
    times.insert(0,0)

    return times

log_key_file = pd.read_csv('../data/System_logs/raw_log_v1.csv')

# define the function for a coming log to match current key log set
def single_match(line, log_key_file):
    variables = []
    log_key = ''
    # get the minimum distance in distances as threshold
    distances = []
    print("we are trying to match %s in our log key set"%(line))
    # reduce the interference of raw log, we will fetch the first half of raw log to compute the distance
    length = len(line)
    # calculate the distance between the given one and log_key_set
    for i in range(len(log_key_file.iloc[0,:])):
        try:
            distance = int(str(brew_distance.distance(line[:int(length*4/5)],log_key_file.iloc[0,:][i]))[1:3])
            distances.append(distance)
        except Exception as error:
            distance = int(str(brew_distance.distance(line[:int(length*4/5)],log_key_file.iloc[0,:][i]))[1:2])
            distances.append(distance)
        print('the %d distance is: %s' % (i, distance))
    threshold = min(distances)

    for j in range(len(log_key_file.iloc[0,:])):
        try:
            distance = int(str(brew_distance.distance(line[:int(length*4/5)],log_key_file.iloc[0,:][j]))[1:3])
        except Exception as error:
            distance = int(str(brew_distance.distance(line[:int(length*4/5)],log_key_file.iloc[0,:][j]))[1:2])
        if distance <= threshold:
            print('the key log for this system log is:',log_key_file.iloc[0,:][j])

            variables = show_diff(line, log_key_file.iloc[0,:][j])
            log_key = log_key_file.iloc[0,:][j]
            break
        else:
            continue
    # we just need to get the variables from this program, as for the time gap,
    # we will create it or just put it directly into the framework

    # process the special case, for which we can find matched system log key
    return variables, log_key


# define the parameter value vector creation function
# we need use the variables in function single_match and append it after the time gap

# define the pattern of system log, we can change it depends on the format of system log
# system_log_pattern = '(\S+) (\w+) (\S+) (\S+) (\S+) (.*)'
# system_log_pattern = '(\S+)(\s+)(\w+) (\S+) (\S+) (\S+) (.*)'
system_log_pattern = '(\S+)(\s+)(\w+)(\s+)(\S+)(\s+)(\S+)(\s+)(\S+)(\s+)(.*)'
# normally, we just analyse the system logs within one day, if we want to analyse logs beyond one day,
# we can extend the time process part
def parameter_vector(filename,time_filename,key_file):
    with open(filename) as f:
        contexts = f.readlines()

    parameter_value_vectors = []
    parameter_value_vector = []
    raw_logs = []
    raw_log = ''
    # this times list is for time gap
    times = []
    # this original_times is for raw log
    original_times = []
    log_keys=[]
    # there is no need to carry out a series of filters and prepossessing here
    # while we do need to use lower() to transfer all the uppercase letters to lowercase, difflib is sensitive
    # generate the time gap
    times = time_gap(time_filename)
    with open('../data/System_logs/times_log.txt') as f:
        original_times=f.readlines()
        for line, time, original_time in zip(contexts, times, original_times):
            match = re.search(system_log_pattern, line)
            if match is None:
                pass
                #raise Exception('Invalid system log is %s'%line)
            else:
                # month = match.group(1)
                # day = match.group(3)
                # time = parse(match.group(5))
                # host_name = match.group(5)
                # syslog_process_uid = match.group(6)
                message = match.group(11).lower()
                try:
                    variables, log_key = single_match(message, key_file)
                    # we write the time gap at index 0
                    # be careful here, do not return the inserted value to a variable
                    variables.insert(0, time)
                    parameter_value_vector = variables

                except Exception:
                    raise
            raw_log = original_time +' '+ message
            raw_logs.append(raw_log.strip('\n'))
            log_keys.append(log_key.strip('\n'))
            print("the parameter_value_vector is:", parameter_value_vector)
            parameter_value_vectors.append(parameter_value_vector)

        # get the time_gaps list
        # for i in range(len(times)):
        #     if i==0:
        #         # we set the t0 == t1
        #         parameter_value_vectors.insert(0,0)
        #     else:
        #         parameter_value_vectors.insert(0,times[i]-times[i-1])
        #

    return raw_logs, log_keys, parameter_value_vectors

# build the framework for log key vector
# refresh the cache of variables
raw_logs, log_keys, parameter_value_vectors = None, None, None
log_key_file = pd.read_csv('../data/System_logs/raw_log_v1.csv')
raw_logs, log_keys, parameter_value_vectors = parameter_vector('../data/System_logs/Linux_2k.log','../data/System_logs/times_log.txt',log_key_file)

print("the raw_logs are:",raw_logs)
print("the log_keys are:",log_keys)
print("the parameter_value_vectors are:",parameter_value_vectors)
fieldnames = ['log message','log key','parameter value vector']
# define the entry dict
entry = {
    'log message':[],
    'log key':[],
    'parameter value vector':[]
}
# transfer the entry to defaultdict
entry = defaultdict(list)

# every row is like: raw log(time + message), log key, parameter value vector
# for the value vector, we should use list.append to add time value, the time value index should be 0
for raw_log, log_key, parameter_value_vector in zip(raw_logs, log_keys, parameter_value_vectors):
    # assign values to every row
    print("we are writing raw_log into log_message:",raw_log)
    entry['log message'].append(raw_log)
    print("we are writing log_key into log_key",log_key)
    entry['log key'].append(log_key)
    print("we are writing parameter_value_vector into dataframe:",parameter_value_vector)
    entry['parameter value vector'].append(parameter_value_vector)


df = pd.DataFrame(entry, columns = fieldnames)
print("the entry is:", entry)
df.to_csv('../data/System_logs/log_value_vector.csv', index=False)

# we can use pandas to transform the timedalta data into int then
# df['tdColumn'] = pd.to_numeric(df['tdColumn'].dt.days, downcast='integer')