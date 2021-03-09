# create the lists belong to different clusters
# we will use the cluster number in last result
from typing import List, Any

from edit_distance import brew_distance_strings
import brew_distance

'''
def cluster_raw_log_keys_lists(filename, cluster_number, threshold_list):
    # we got the threshold is 16 for the clustering in log_key.ipynb
    # if the distance weight is below the threshold, we will move the two compared logs to a group
    threshold = 16
    cluster_lists = []
    # here, threshold0 = threshold
    # create different lists of clusters
    cluster_list = [[] for i in range(cluster_number)]
    # for i in range(cluster_number):
    #    cluster_number + 'i'= []

    with open(filename, 'r') as f:
        # we should build a recursive function to do that
        contexts = f.readlines()
        for n in range(len(cluster_list) - 1):
            for i in range(len(contexts) - 1):
                for j in range(len(threshold_list)):
                    try:
                        print("Compare two raw logs!")
                        print(contexts[i])
                        print(contexts[i + 1])
                        # print(str(brew_distance.distance(contexts[i], contexts[i+1]))[1:3])
                        distance = str(brew_distance.distance(contexts[i], contexts[i + 1]))[1:3]
                    except brew_distance.BrewDistanceException as error:
                        print(str(error))

                    print("the distance is:", distance)

                    # the process of comparation

                    if int(distance) <= threshold_list[j]:
                        cluster_list[n].append(contexts[i])
                        cluster_list[n].append(contexts[i + 1])
                    elif int(distance) <= threshold_list[j + 1]:
                        cluster_list[n].append(contexts[i])
                        cluster_list[n + 1].append(contexts[i + 1])
        cluster_lists.append(cluster_list[n])
        for i in range(len(cluster_lists)):
            print("the cluster %d is:" % i, cluster_lists[i])

        return cluster_lists


cluster_raw_log_keys_lists('data/System_logs/lower_erased_message_log.txt', 3, [16, 29, 40])


'''

# def weights_pair_dictionary(filename):
#     # first, we should make a dictionary, with the key of weights and the corresponding lines
#     weights_pair_dict = {}
#     with open(filename, 'r') as f:
#         contexts = f.readlines()
#         for i in range(len(contexts)-1):
#             distance = str(brew_distance.distance(contexts[i], contexts[i + 1]))[1:3]
#             weights_pair_dict[distance] = contexts[i], contexts[i+1]
#
#         # print(weights_pair_dict)
#         return weights_pair_dict
#
#
# weights_pair_dict = weights_pair_dictionary('data/System_logs/lower_erased_message_log.txt')
#
# thresholds = [16,29]
# weights = [40, 13, 26, 29, 12, 16, 12, 26, 29, 12, 29, 29]
# number_cluster=3


'''
def group_logs(dict, number_cluster, thresholds):
    cluster_list = [[] for i in range(number_cluster)]
    weights_pair_dict = dict
    # weithts = weights.sort()
    for i in range(len(thresholds)-1):
        for n in range(number_cluster):
            for key, value in weights_pair_dict.items():
                if int(key) < thresholds[i]:
                    cluster_list[n].append(weights_pair_dict[key])
                elif cluster_list[n] is None:
                    cluster_list[n].append(weights_pair_dict[key][0])
                    cluster_list[n + 1].append(weights_pair_dict[key][1])
                else:
                    # get the first value of pair
                    # weights_pair_dict[key][0]
                    distance = str(brew_distance.distance(weights_pair_dict[key][0], cluster_list[n][0]))[1:3]
                    if int(distance) < threholds[i]:
                        cluster_list[n].append(weights_pair_dict[key][0])
                    elif int(distance) < threholds[i+1]:
                        cluster_list[n+1].append(weights_pair_dict[key][0])
                    else:
                        cluster_list[n+2].append(weights_pair_dict[key][0])
    print(cluster_list)
    return cluster_list

weights_pair_dict = weights_pair_dictionary('data/System_logs/lower_erased_message_log.txt')
group_logs( weights_pair_dict, number_cluster=3, thresholds=[16,29])

'''


# this is code for test with mini number
'''
def group_logs(dict, number_cluster, thresholds):
    cluster_list = [[] for i in range(number_cluster)]
    weights_pair_dict = dict
    # weithts = weights.sort()
    for i in range(len(thresholds)-1):
        for n in range(number_cluster-2):
            for key, value in weights_pair_dict.items():
                if int(key) < thresholds[i]:
                    cluster_list[n].append(weights_pair_dict[key])
                elif cluster_list[n] is None:
                    cluster_list[n].append(weights_pair_dict[key][0])
                    cluster_list[n + 1].append(weights_pair_dict[key][1])
                elif:
                    distance = str(brew_distance.distance(weights_pair_dict[key][0], cluster_list[n][0]))[1:3]
                    if int(distance) < threholds[i]:
                        cluster_list[n].append(weights_pair_dict[key][0])
                    elif int(distance) < threholds[i+1]:
                        cluster_list[n+1].append(weights_pair_dict[key][0])
                    else:
                        cluster_list[n+2].append(weights_pair_dict[key][0])
    print(cluster_list)
    return cluster_list

weights_pair_dict = weights_pair_dictionary('data/System_logs/lower_erased_message_log.txt')
group_logs( weights_pair_dict, number_cluster=3, thresholds=[16,29])
'''

'''
def group_logs(dict, number_cluster, thresholds):
    cluster_list = [[] for i in range(number_cluster)]
    weights_pair_dict = dict
    for key, value in weights_pair_dict.items():
    # initialize the cluster_list:
    
        if int(key) < thresholds[0]:
            print("the key is:",int(key))
            cluster_list[0].append(value)
        elif int(key) < thresholds[1]:
            cluster_list[0].append(weights_pair_dict[key][0])
            cluster_list[1].append(weights_pair_dict[key][1])
        else:
            cluster_list[0].append(weights_pair_dict[key][0])
            cluster_list[1].append(weights_pair_dict[key][1])
    print("cluster_list[0] is:",cluster_list[0])
    print("cluster_list[1] is:",cluster_list[1])
    return cluster_list[0], cluster_list[1]

weights_pair_dict = weights_pair_dictionary('data/System_logs/lower_erased_message_log.txt')

group_logs(weights_pair_dict, number_cluster =3, thresholds = [16,29])
'''


'''
we will use basic recursive function to do:
the following is the basic principle about recursive funtion:

def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)
the variable in this program should be the order of cluster

'''

'''
import numpy as np

def group_logs_cluster_initial(filename,cluster_number, threshold):

    cluster_list = [[] for i in range(cluster_number)]
    distances = []

    with open(filename,'r') as f:
        contexts = f.readlines()

    for i in range(len(contexts)):
        if cluster_list[0] is not None:
            break
        else:
            for j in range(len(contexts)-i):
                try:
                    # print(str(brew_distance.distance(contexts[i], contexts[i+1]))[1:3])
                    distance = int(str(brew_distance.distance(contexts[i], contexts[j]))[1:3])
                    # raise Exception('transformation error')
                    distances.append(distance)
                except Exception as error:
                    pass

                # we are gonna choose the base cluster for grouping clusters
                # distances is the list of distance

            distances = np.asarray(distances)
            print('the distance is:', distances)
            results = (distances <= threshold)
            print("the compared results are:", results)
            for i in range(len(results)):
                if results[i] == 'Flase':
                    results[i] = 0
                else:
                    results[i] = 1
            results = list(results)
            if results.count(1) > len(contexts)//cluster_number:
                cluster_list[0].append(contexts[i])
                break
            else:
                continue
        print(cluster_list[0])
        print(len(cluster_list[0]))

        return cluster_list[0]

group_logs_cluster_initial('data/System_logs/lower_erased_message_log.txt',cluster_number = 3, threshold = 16)

#def group_logs(cluster_list[0],lines):
'''


# 3 is the cluster_number here ----- for small testing dataset
# cluster_list = [[] for i in range(3)]

# we get the cluster number is 44 from log_key.ipynb file ----- for large dataset
# the total should be 94, while the final 3 clusters are None, so we use number 91 finally
cluster_list = [[] for i in range(91)]
print("the cluster_list is:",cluster_list)
f = open('../data/System_logs/lower_erased_message_log.txt','r')
lines = f.readlines()

#distance_test = str(brew_distance.distance("session opened for user cyrus by (uid", "session opened for user news by (uid"))

#print(distance_test)

def cluster_create (lines, cluster_number, threshold, cluster_list):
    # the list for unable processed system logs
    special_lines = []

    print("the current lines are:",lines)
    print("the length of current lines is:", len(lines))
    #print(type(lines))
    # print(type(cluster_list))

    if lines is not None and lines != [] and cluster_number >= 0:

        #print(cluster_number)
        print('the cluster_index is:',cluster_number)

        cluster_list[cluster_number].append(lines[0])
        #print('the cluster_list is:',cluster_list[cluster_number])

        for i in range(1,len(lines)-1):
            try:
                distance = int(str(brew_distance.distance(lines[0], lines[i]))[1:3])
            except Exception as error:
                distance = int(str(brew_distance.distance(lines[0], lines[i]))[1:2])
        # print('the current distance is:',distance)
            if distance < threshold:
                cluster_list[cluster_number].append(lines[i])
        print("the cluster_list[%d] is: %s"%(cluster_number, cluster_list[cluster_number]))

        # print(len(cluster_list[cluster_number]))
        # print(type(cluster_list[cluster_number]))

        # lines.remove(cluster_list[cluster_number])


        for j in range(len(cluster_list[cluster_number])):
            # print(len(cluster_list[cluster_number]))
            # print(type(cluster_list[cluster_number]))
            # remove the grouped lines from current lines
            lines.remove(cluster_list[cluster_number][j])
            # we append the logs that are unable to process to special_lines as single key logs
            # print('the removed lines are:',lines)
            # delete the lines grouped already
            # print('the current lines are:', lines)
            # print('the appended cluster_list[cluster_number] is:',cluster_list[cluster_number])
			# print("the length of cluster_list[number] is",len(cluster_list[cluster_number]))
        return cluster_create(lines=lines, cluster_number=cluster_number-1, threshold=threshold, cluster_list=cluster_list)
    # 0 means the final lines that have been removed the grouped parts
    else:
        # the special_lines are what we use to evaluate the performance of grouping clusters
        # until the value is None, we can get the accurate cluster number
        special_lines.append(lines)
        '''
        with eps=0.01 min_samples=1 ----- we got 50 in the end
        so we will add 50 more clusters to the total clusters to get those involved in 
        ---- the total should be 94, while the final 3 clusters are None, so we use number 91 finally
        '''
    return cluster_list, special_lines


# the value of cluster_number here should be the cluster_number minus 1, we use 33-1=32 for large dataset
#print(cluster_create(lines=lines, cluster_number=38, threshold=12, cluster_list=cluster_list))
cluster_list, speical_lines = cluster_create(lines = lines, cluster_number=90, threshold=11, cluster_list = cluster_list)
print("the cluster_list is:", cluster_list)
print("the unable processed lines are:", speical_lines)


'''
when we set threshold to 16:
    we find a wired one cluster: zapping low mappings\n', ' probing pci hardware\n', ' scanning for pnp cards\n

after get this, we try to decrease the threshold to 11 to see the performance:
    the cluster_list[30] is: [' session opened for user cyrus by uid\n'
    the cluster_list[28] is: [' session closed for user news\n',
    
    according to the paper: the above two kinds of raw logs definitely should be separated, which is not as expected 
    by experience
     
    
'''