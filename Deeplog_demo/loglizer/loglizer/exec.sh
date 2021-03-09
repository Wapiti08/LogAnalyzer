#!/bin/bash

path1='../../Dataset_ML/Linux/Malicious/structured_log.csv'
path2='../../Dataset_ML/Linux/Malicious/Event_dict.pkl'
path3='../../Dataset_ML/Linux/Malicious/structured_log_id.csv'
path4='../../Dataset_ML/Linux/Malicious/Linux_matrix/log_matrix.npy'
path5='../../Dataset_ML/Linux/Clear/structured_log.csv'
path6='../../Dataset_ML/Linux/Clear/Event_dict.pkl'
path7='../../Dataset_ML/Linux/Clear/structured_log_id.csv'
path8='../../Dataset_ML/Linux/Clear/Linux_matrix/log_matrix.npy'

python3 matrixgen.py --p1 $path1 --p2 $path2 --p3 $path3 --p4 $path4 --p5 $path5 --p6 $path6 --p7 $path7 --p8 $path8

exit 0
