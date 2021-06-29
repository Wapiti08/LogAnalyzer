#!/bin/bash

path1="../Dataset/Linux/Client/Client_structured/Integrated_structured_log.csv"
path2="../Dataset/Linux/Client/Client_structured/log_value_vector.csv"
path3="../Dataset/Linux/Client/Client_structured_com/Integrated_structured_log.csv"
path4="../Dataset/Linux/Client/Client_structured_com/log_value_vector.csv"

python3 Execution_Path_Anomaly_para.py --p1 $path1 --p2 $path2 --p3 $path3 --p4 $path4



#echo ../Dataset/Linux/Client/Client_structured/Integrated_structured_log.csv
#echo ../Dataset/Linux/Client/Client_structured/log_value_vector.csv
#echo ../Dataset/Linux/Client/Client_structured_com/Integrated_structured_log.csv
#echo ../Dataset/Linux/Client/Client_structured_com/log_value_vector.csv
