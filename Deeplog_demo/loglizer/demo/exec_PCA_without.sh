#!/bin/bash

path1='../../Dataset_ML/Linux/Malicious/Linux_matrix/log_matrix.npy'
path2='../../Dataset_ML/Linux/Clear/Linux_matrix/log_matrix.npy'

python3 PCA_demo_without_labels.py --p1 $path1 --p2 $path2

exit 0
