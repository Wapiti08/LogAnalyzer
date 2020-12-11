# DeepLog

![Authour](https://img.shields.io/badge/Author-Wapiti08-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.7-brightgreen.svg) 
![RNN](https://img.shields.io/badge/RNN-GRU-redgreen.svg)
![Analysis](https://img.shields.io/badge/Analysis-Anomaly%20logs-redgreen.svg)
![License](https://img.shields.io/badge/license-MIT3.0-green.svg)

---

- This is the achievement of core DeepLog for research aim.

- It is the basic thought with feature engineering to analyse raw logs and finally report the potential malicious logs based on a series of processings.

- The **Online Update** part for models please check [Online Update](https://gist.github.com/Wapiti08/d47787beb01cbb5777bdf655cfffef64)

## Feature:

- convert the logs to structured pandas framework
- extract the log keys from raw logs
- analyse the log key exeuction path
- analyse the paramaters in log key
- combine results from both model
- analyse the time series data generated from window size and time interval by PCA. 

For the dataset, I have given some examples and you can put your own data into that folder.

## pre-preparation:

```
# in order to match the libraries versions, please run and build the project in virtual environment
virtualenv env
pip3 install -r requirement.txt
```

## Instructions (In Deeplog_demo folder):

###  1. Source data:
When the data format is in csv, we need translate them into txt files and split them into batches.
```
python3 csv_txt_trans.py 
```
You will get notice on inputing the source location and output location.

###  2. Data analysis:
we use the logparser tool to transform the source txt log files into structured csv files under a folder, the folder is named by the start and end time. (Find the Lenma_demo under the logparser/logparser/demo)

**(use Lenma_demo.py with python2)** ---> The python3 version is not provided here.
You need to set the locations first:
```
input_dir = '../../Dataset/Linux/Clear/'   # set the location to yours
output_dir = '../../Dataset/Linux/Clear_Separate_Structured_Logs/'    # set the location to yours
```
Then you can execute the demo file with python 2.x:
```
python Lenma_demo.py 
```

In the stage, we calculate the EventTemplate for every log. 

###  3. Variable Selection:
The log_value_vector.py will be used to generate the csv file, which will be used to implement the anomaly detection later. 

![Parameter_vector.png](https://github.com/Wapiti08/DeepLog/blob/master/Deeplog_demo/Pic/Dataframe.png)



**(and has been integrated into models already in demo)**

###  4. Model detection:
Basiclly, we have two modules for DeepLog 

- Whereas, before implementing the modules, we will first see whether there is obvious malicious logs, we will report them first.

- After that, we will first implement execution path anomaly detection with Execution_Path_Anomaly.py
	
- Finally, we will implement parameter values anomaly detection with Parameter_value_performance_anomaly.py	

- As a plus, there is the ML model using PCA in loglizer.

For basic instructions, please also check the ![Deeplog_datafrom.png](https://github.com/Wapiti08/DeepLog/blob/master/Deeplog_demo/Deeplog_dataflow.png).

## Statement:
- The model is based on off-line work, the online real-time detection is not available.
- The [loglizer](https://github.com/logpai/loglizer) and [logparser](https://github.com/logpai/logparser) are open source tools, author's rights are reserved.
- I enriched the two tools in the project, notice the differences from the original version.

## References：
*1.Execution Anomaly Detection in Distributed Systems through Unstructured Log Analysis*

*2.DeepLog: Anomaly Detection and Diagnosis from System Logs*

*3.Incremental Construction of LSTM Recurrent Neural Network*
