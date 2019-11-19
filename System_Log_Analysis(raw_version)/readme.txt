# this is the instruction how to use the a series of modules to pre-process the raw system logs to key logs

all the inspiration is from the article: Execution Anomaly Detection in Distributed
                                         Systems through Unstructured Log Analysis

statement:
the series of programs are composed of .ipynb and .py files. ipynb is easier to carry out block test while some libraries
can not be download through conda. So we use pycharm to finish the parts that jupyter can not finish

1. we get raw system logs first:
the format is: Jun 16 04:10:22 combo su(pam_unix)[25178]: session opened for user cyrus by (uid=0)
   1. we only need message and timestamp parts.
   we use parse_system_log_line function in “log_key.ipynb” to get times_log.txt and messages_log.txt files

2. then we process the messages_log.txt:
   1. we use erase_string function in "log_key.ipynb" to replace some special characters
   2. use transfer_lowercase function to transfer all uppercase letters into lowercase ones
   3. use transform_to_csv function to transfer the txt message file into csv file

3. calculate the weights between two processed logs
   use edit_distance.py program to calculate the distances between two logs and return a log_distance.txt file
   we use edit_distance to get the threshold for inner-class cluster, and combine K-Means function in log_key.ipynb
   get the threshold is 16.
   we extend the cluster algorithm to DBSCAN and automatic create the clusters(we use this one for our process)

4. Clustering the raw logs into different clusters according to the threshold
    use cluster_raw_log_keys_lists.py to get the clusters of raw logs. the comment parts are the tests we did before.
    we used many different process methods. The final part is most efficient one.

5. Splitting the clustered logs
    1. In order to make the process easily, we made "cluster_list.csv" and "common_words.csv" files with cluster_list_csv
    function and common_words_csv function in log_splitting.py.ipynb file.
    2. we use the common_words for a cluster to split the cluster_list with csv_privates and split_key_csv functions in
    log_splitting.py.ipynb file. we get the framework for privates.csv file.
    3. we use batch_processing_logs function to process one cluster and get complete privates.csv for this group
    4. we use parameters_key_log to find the parameters((the different number in the same position is equal or above 4))
    in a key log
    5. use raw_log_create to delete parameters in key log and get the finally key logs