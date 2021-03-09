#!/bin/bash
echo the path of key_para_dict.csv and tokens_dict.pkl is default
echo you can change it to your own directory

echo the root_dir is default, you can modify it according to your situation

path1="../Dataset/Linux/Malicious_Separate_Structured_Logs/log_value_vector.csv"
path2="../Dataset/Linux/Malicious_Separate_Structured_Logs/key_num_para_dict.csv"
path3="../Dataset/Linux/Malicious_Separate_Structured_Logs/Event_npy/"

python3 Parameter_value_vector_anomaly.py --p1 $path1 --p2 $path2 --p3 $path3
