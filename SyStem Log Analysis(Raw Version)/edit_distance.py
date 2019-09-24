import brew_distance

global distance_size

# filename is the file of prepocessed log file, filename1 is the file of distance list
# def brew_distance_strings(filename,filename1):
#     with open(filename, 'r') as f:
#         with open(filename1, 'w') as f1:
#             contexts = f.readlines()
#             #print(type(contexts))
#             list_distance = ''
#             for i in range(len(contexts)-1):
#
#                 try:
#                     print("Determing two raw logs!")
#                     print(contexts[i])
#                     print(contexts[i+1])
#                     # print(str(brew_distance.distance(contexts[i], contexts[i+1]))[1:3])
#                     list_distance += str(brew_distance.distance(contexts[i], contexts[i+1]))[1:3]
#                 except brew_distance.BrewDistanceException as error:
#                     print(str(error))
#             print(list_distance)
#             list = list_distance.split(' ')
#             distance_size = len(list) -1
#             print(distance_size)
#             for i in range(len(list)):
#                 print(list[i])
#                 f1.write(list[i])
#                 f1.write('\n')
#             return list
#
# brew_distance_strings('../data/System_logs/lower_erased_message_log.txt','../data/System_logs/log_distance.txt')
# the value is calculated by log_key.ipynb

# import numpy as np
# print("the distance between string is:")
# string1 = 'PCI: Using IRQ router PIIX/ICH [8086/2410] at 0000:00:1f.0'
# string2 = 'PCI: Probing PCI hardware (bus 00)'
# a = str(brew_distance.distance(string1, string2))[1:3]
# print(a)
# string1_np = np.array(string1)
# string2_np = np.array(string2)

# print("the distance between ndarray is:")


# b = str(brew_distance.distance(string1_np, string2_np))[1:3]

string1 = 'authentication failure: logname= uid=0 euid=0 tty=nodevssh ruser= rhost=220-135-151-1.hinet-ip.hinet.net user=root'
string2 = 'real time clock driver v'
string3 = 'authentication failure logname uid euid tty ruser rhost'
print('the length of string1 is:',len(string1))
length = len(string1)
print("the part of string1 is:",string1[:int(length*4/5)])
distance_1 = str(brew_distance.distance(string1[:int(length*4/5)],string2))[1:3]
distance_2 = str(brew_distance.distance(string1[:int(length*4/5)],string3))[1:3]
print(distance_1)
print(distance_2)

string4 = 'cupsd startup succeeded'

string7 = 'cupsd shutdown succeeded'

string5 = 'restart'
string6 = '1.4.1: restart'

distance_1 = str(brew_distance.distance(string4[:int(length*4/5)],string5))[1:3]
distance_2 = str(brew_distance.distance(string6[:int(length*4/5)],string5))[1:3]
print(distance_1)
print(distance_2)
