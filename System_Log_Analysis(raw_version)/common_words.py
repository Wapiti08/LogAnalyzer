# this is the part about splitting the group logs
# There is another threshold to distinguish parameters and the part of log keys:
# if is bigger than the threshold y, they are parameters ------ y is tested to be 4
## 1. we need to get the common word sequence of raw log keys ---- CW1, CW2, ... ,CWn
## 2. the word sequence divides ith log key into DW1(i),DW2(i),DW3(i),.... in total (N+1) parts
##  DWj(i) is the private content at position j of the ith raw log key
## 3. for position j, obtain GN private contents at position j from GN raw log keys, they are DWj(1),DWj(2),...
## 4. donate number of "different" values among those GN values as VNj,
## If VNj is equal to or bigger than a threshold a,---- parameters
## Entropy rules, called EPj, if the value is small, the private contents at that position --- parts of log keys


from sklearn.feature_extraction.text import CountVectorizer
from cluster_raw_log_keys_lists import cluster_create
import csv

word_threshold = 4
# load the data
f = open('../data/System_logs/lower_erased_message_log.txt','r')
lines = f.readlines()
print("-----------------------------")
print("lines are:", lines)

# the number in range should be specified for different dataset
cluster_list = [[] for i in range(91)]

cluster_list, speical_lines = cluster_create(lines = lines, cluster_number=90, threshold=11, cluster_list = cluster_list)
# cluster_list = cluster_create(lines=lines, cluster_number=2, threshold=16, cluster_list=cluster_list)
print("------------------------------------------")
print("the cluster_list is:",cluster_list)

'''
we should notice that:
    the common word sequence is for a group
    we should separately calculate the word sequence
'''

def top_freq_words(cluster_lines, n):

    words_bag = []

    # get the bag of word model that has removed special characters
    vec = CountVectorizer().fit(cluster_lines)
    # print("the vec is",vec)
    # print("the vec vocabulary is:", vec.vocabulary_)
    # print("the vec vocabulary items are:", vec.vocabulary_.items())

    '''
      (0, 6)	1
      (0, 8)	1
      (0, 10)	1
      (0, 13)	1
      (0, 14)	1
      (0, 16)	1
      (0, 17)	1
    '''
    # get the bag_of_words matrix, bag_of_words[i,j] is the occurrence of word j in the text i
    bag_of_words = vec.transform(cluster_lines)
    # print("the bag_of_words are:", bag_of_words)
    # calculate the sum, axis=0 means the column
    sum_words = bag_of_words.sum(axis=0)
    # print("the sum_words are:",sum_words)
    # get the words_freq
    words_freq = [(word, sum_words[0,idx]) for word, idx in vec.vocabulary_.items()]
    print("words_freq:", words_freq)
    # sort the words_freq
    words_freq = sorted(words_freq, key = lambda x:x[1], reverse = True)
    # print("sorted words_freq are:", words_freq)
    highest_freq = words_freq[0][1]
    # print("the wrod is:",n)
    print("the highest_freq is:",highest_freq)
    print(words_freq[:n])


    return words_freq[:n], highest_freq


def common_word_sequene(words_freq, highest_freq):

    common_words = []
    for item in words_freq:
        if item[1] == highest_freq:
            common_words.append(item[0])
    # for i in range(len(words_freq)):
    #     common_words.append(words_freq[i][0])

    print(common_words)
    return common_words

#print("the top freq of words are:",words_freq[])
for i in range(len(cluster_list)):
    cluster_lines = cluster_list[i]
    print("------------------------------------------------")
    print("the cluster_list %d is %s:"%(i, cluster_list[i]))
    # n should be set here to extract the most common key
    # words according to the most frequency words in every group
    # words_freq, highest_freq = top_freq_words(cluster_lines=cluster_list[i], n=100)
    words_freq, highest_freq = top_freq_words(cluster_lines=cluster_lines, n=100)
    common_words = common_word_sequene(words_freq=words_freq, highest_freq=highest_freq)

# I want to build two csv files:
## 1. cluster_list order with every key log in this cluster_list
## 2. the cluster_list order with the corresponding common_words

# build the cluster_list csv file
import pandas as pd


def cluster_list_csv(cluster_list):
    # use pandas to write data
    dict_cluster = {}
    cluster_list_names = []

    lengths = []
    for n in range(len(cluster_list)):
        print("the length of cluster_list %d is %d" % (n, len(cluster_list[n])))
        lengths.append(len(cluster_list[n]))
    print("the max length is", max(lengths))
    # get the max length of cluster_list
    max_length = max(lengths)

    for i in range(len(cluster_list)):
        cluster_list_names.append('cluster' + str(i))

    for i in range(len(cluster_list)):
        if len(cluster_list[i]) == max_length:
            pass
        else:
            cluster_list[i] = cluster_list[i] + [''] * (max_length - len(cluster_list[i]))
        dict_cluster['cluster' + str(i)] = cluster_list[i]

    print("the dict_cluster is:", dict_cluster)
    print("the cluster_list_names are:", cluster_list_names)
    df = pd.DataFrame(dict_cluster, columns=cluster_list_names)
    df.to_csv('../data/System_logs/cluster_list.csv',index = False)
    print(df)

cluster_list_csv(cluster_list)


# build the common_words csv file
def common_words_csv(cluster_list):
    # write the common word into a csv file using pandas
    # use pandas to write data
    dict_common_words = {}
    cluster_list_names = []

    # get the max length of cluster_list
    lengths = []
    for k in range(len(cluster_list)):
        words_freq, highest_freq = top_freq_words(cluster_lines=cluster_list[k], n=100)
        common_words = common_word_sequene(words_freq=words_freq, highest_freq=highest_freq)
        # common_words = common_words.remove('none')
        print("the length of common_words for cluster_list %d is %d" % (k, len(common_words)))
        lengths.append(len(common_words))
    print("--------------------this is boundary-------------------------")
    print("the max length is", max(lengths))

    max_length = max(lengths)


    for i in range(len(cluster_list)):
        cluster_list_names.append('cluster' + str(i))

    for k in range(len(cluster_list)):
        words_freq, highest_freq = top_freq_words(cluster_lines=cluster_list[k], n=100)
        common_words = common_word_sequene(words_freq=words_freq, highest_freq=highest_freq)

        # check whether there are same lengths
        if len(common_words) == max_length:
            pass
        else:
            common_words = common_words + [''] * (max_length - len(common_words))

        dict_common_words['cluster' + str(k)] = common_words

    df = pd.DataFrame(dict_common_words, columns=cluster_list_names)
    df.to_csv('../data/System_logs/common_words.csv',index=False)
    print(df)


    # write the common_words into the csv file using csv library
    # cluster_list_names = []
    # with open('../data/System_logs/common_words.csv', 'w') as f:
    #     csvfile = csv.writer(f)
    #     # write the column names
    #     for i in range(len(cluster_list)):
    #         cluster_list_names.append('cluster_list' + str(i))
    #     csvfile.writerow(cluster_list_names)
    #     # get the common words for every cluster_list
    #     for j in range(len(cluster_list)):
    #         words_freq, highest_freq = top_freq_words(cluster_lines=cluster_list[j], n=100)
    #         common_words = common_word_sequene(words_freq=words_freq, highest_freq=highest_freq)
    #         print("the common words for clsuter_list %d are: %s"%(j, common_words))
    #         csvfile.writerow({'cluster_list'+str(j):common_words})

common_words_csv(cluster_list)



