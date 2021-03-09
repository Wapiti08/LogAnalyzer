#!/usr/bin/env python
# coding: utf-8

# In[84]:


'''
This is used for finding the log pattern in a log file
'''
# filter the first parts except the message part
# read time to other part
# in this program, spark will be used to calculate the key with collect
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import re
from pyspark.sql import Row

global patterns
system_log_pattern = '(\S+) (\w+) (\S+) (\S+) (\S+) (.*)'


# the module to read file:
'''
def read_log_file(filename):
    lines,times = [],[]
    with open(filename,'r') as f:
        for line in f:
            print(type(line))
            #line = line.split(' ')
            time = line[2:3]
            line = line[5:]
            lines.append(line)
            times.append(time)
        return times,lines
'''
'''
# this is module used to process the json file log
def parse_system_log(filename):
    with open(filename, 'r') as f:
        for line in f:
            match = re.search(system_log_pattern, line)
        if match is None:
            raise Exception ('Invalid system log %s'%line)
        return Row(
            month = match.group(1),
            day = match.group(2),
            time = match.group(3),
            host_name = match.group(4),
            syslog_process_uid = match.group(5),
            message = match.group(6)
        )
'''


# this is the module used to process the lined log file
def parse_system_log_line(filename):
    with open(filename,'r') as f:
        with open('data/System_logs/times_log.txt','w') as f1:
            with open('data/System_logs/messages_log.txt','w') as f2:
#                 times = ''
#                 messages = ''
                for line in f:
                    match = re.search(system_log_pattern, line)
                    if match is None:
                        raise Exception ('Invalid system log %s'% line)
                    else:
                        month = match.group(1)
                        day = match.group(2)
                        time = match.group(3)
                        host_name = match.group(4)
                        syslog_process_uid = match.group(5)
                        message = match.group(6)

                        f1.write(time + '\n')

                        f2.write(message + '\n')


# read filtered lines into two file, time_file and message_file
# def write_basic_log_keys(times, lines):
#     with open("data/times_log.txt",'w') as f1:
#         for i in range(len(times)):
#             f1.write(str(times[i]))
        
#     with open("data/messages_log.txt",'w') as f2:
#         for j in range(len(lines)):
#             f2.write(str(lines[j]))


import re

# this part is used for erasing parameters by empirical rules
def erase_string(filename):
    # the file should be messages_log.txt
    with open(filename,'r') as f:
        with open('data/erased_message_log.txt','w') as f1:
            for line in f:
                line = line.split(' ')    
                for i in range(len(line)):
                    line[i] = line[i].split('=')[0]
                    
                line = list(line)
                # filter the blank in a list, and line_str will be string
                line_str = ' '.join(line).split()
                line_list = list(line_str)
                # transfer the list data to string and implement further filter
                line_str = ' '.join(line_list)
                # file all the numbers
                line_str = re.sub("\d|;|[|]|(|)|=",'',line_str)
                print(line_str)
                #line_str = line_str.strip()  
                #print(type(line_str))
                print(line_str)
                f1.write(line_str+'\n')


def compose_earse_string(filename1, filename2):
    with open(filename1, 'r') as f:
        line_str = ''
        with open(filename2, 'w') as f1:
            for line in f:
                print(type(line))
                print(line+'\n')
                for i in range(len(line)):
                    line_str += line[i]
                    f1.write(line_str)
    

compose_earse_string('data/System_logs/erased_message_log.txt','data/System_logs/composed_message_log.txt')

'''

def transfer_lowercase(lines):
	for line in lines:
		for i in range(len(line)):
            line[i]=line[i].lower()
            if ';' in line[i]:
                line[i]=line[i].strip(':')
                for j in range(i):
                    patterns.append(line[j]+=' '+line[j])
                break
            else:
                continue
			
	return lines


# log_analyzer_sql
# build sc example
conf = SparkConf().setAppName('Log Analyzer SQL')
sc = SparkContext(conf = conf)
sqlContext = SQLContext(sc)

# define the path of logFile
logFile = 'data/System_logs/Linux_2k.log'

# build the spark template
# create the persist cache then we can use it later
system_logs = (sc.textFile(logFile).map(parse_system_log).cache())
schema_system_logs = sqlContext.createDataFrame(system_logs)
# register the data tempalte and name it
schema_system_logs.registerTemplate('logs')



# implement statistical analysis


# this is the part to cluster the keys
# we need to transfer the list into a whole string
with open('data/System_logs/erased_message_log.txt','r') as f:
    for line in f:
        print(type(line))
        print(line[:30])


transfer_lowercase(lines)


def pattern_log(lines_lower):
	
	input: lines

	process: define new patterns list, find the variable(numerical one)
			set the variable to *, append the found new patterns to patterns

			special notice:
			if the initial x chars for two strings are the same, we can ensure the two strings are the same patterns 
	
    output: found patterns

	we need add separated parts to one string:
		use string+' '+string1+...
	
    patterns=[]
	for i in range(len(lines)-1):
        
	# this is the test for find the different part:

# 	for i in range(1,len(lines_lower)-1):
# 		result = []
# 		for a, b in zip(lines_lower[i], lines_lower[i+1]):
# 			if a != b:
# 				break
# 			result.append(a)
# 	return ''.join(result)
        
        # if there is ';' exists, we will fetch the part before that sign as k
        
        # we define the first 5 words are the identifier for different patterns
        
        if lines[i][:5]==lines[i+1][:5]:
            # find the different parts of the same pattern,
            # fetch the first part as the variable and
            # transfer the textual part to numerical part
            dif_part=[]
            
            for j in range(min(len(lines[i]),len(lines[i+1]))):
                
                if lines[i][j]!=lines[i+1][j]:
                    try:
                        int(lines[i][j])
                    except Exception as error:
                        print(error)
                        print('textual data will be transfered to int data')
                        lines[i][j]='*'
                        # we only need to define one variable,this is the sign for us to stop the search
                        dif_part.append(lines[i][j])
                        break
                else:
                    while not len(dif_part):
                        continue
                patterns.append()


lines = read_log_file('data/Linux_10.txt')
lines = transfer_lowercase(lines)
pattern_log(lines)


if __name__=='__main__':
	lines = read_log_file('data/Linux_10.txt')
	lines = transfer_lowercase(lines)
	result = pattern_log(lines)
	print(result)



'''

