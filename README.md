# DeepLog
- This is the achievement of core DeepLog for research aim.

- The **Online Update** part for models is waiting to go....

## Function:

**It is the thought on how to use the a series of modules to pre-process the raw system logs to key logs and report the potential malicious logs.**

For the dataset, I have given some examples and you can put your own data into that folder.


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

## Addition:
If you want to implement the raw version, please run the following command before you go:
```
pip install -r requirement.txt
```

## Statement:
- The model is based on off-line work, the online real-time detection is not available.
- The [loglizer](https://github.com/logpai/loglizer) and [logparser](https://github.com/logpai/logparser) are open source tools, author's rights are reserved.
- I enriched the two tools in the project, notice the differences from the original version.

## References：
*1.Execution Anomaly Detection in Distributed Systems through Unstructured Log Analysis*

*2.DeepLog: Anomaly Detection and Diagnosis from System Logs*

*3.Incremental Construction of LSTM Recurrent Neural Network*
