#!/bin/bash

echo copying the structured csv to Dataset_ML to analyse the logs with Machine Learning Model

cp ../../Dataset/Linux/Malicious_Separate_Structured_Logs/Integrated_structured_log.csv ../../Dataset_ML/Linux/Malicious/structured_log.csv

cp ../../Dataset/Linux/Clear_Separate_Structured_Logs/Integrated_structured_log.csv ../../Dataset_ML/Linux/Clear/structured_log.csv
