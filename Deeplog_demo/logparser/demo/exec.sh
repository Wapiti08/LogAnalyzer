#!/bin/bash

path1="../../Dataset/Linux/Malicious/"
path2="../../Dataset/Linux/Malicious_Separate_Structured_Logs/"

python2 LenMa_demo.py --p1 $path1 --p2 $path2
