#!/bin/bash

path1="../Dataset/Linux/Malicious_Separate_Structured_Logs/Integrated_structured_log.csv"
path2="../Dataset/Linux/Malicious_Separate_Structured_Logs/log_value_vector.csv"
path3="../Dataset/Linux/Clear_Separate_Structured_Logs/Integrated_structured_log.csv"
path4="../Dataset/Linux/Clear_Separate_Structured_Logs/log_value_vector.csv"


python3 Execution_Path_Anomaly.py --p1 $path1 --p2 $path2 --p3 $path3 --p4 $path4
