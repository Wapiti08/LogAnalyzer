# DeepLog
This is the achievement of core DeepLog

For privacy consideration, the dataset is not provided. You can put your own data into that folder

# this is the instruction how to use the a series of modules to pre-process the raw system logs to key logs

all the inspiration is from the two articles: 
	1.Execution Anomaly Detection in Distributed Systems through Unstructured Log Analysis
	2.DeepLog: Anomaly Detection and Diagnosis from System Logs

## Statement:
Currently, the model is based on off-line work, we will achieve online real-time detection in the future

The following is the instructions on how to implement the whole module:
##  1. Source data:
	the data got from clients is all csv format, we need translate them into txt format files.

##  2. Data analysis:
	we use the logparser tool to transform the source txt log files into structured csvs under a folder, the folder is named by the start and end time
	*(use Lenma_demo.py with python2)
	In the stage, we calculate the EventTemplate for every log

##  3. Variable Selection:
	Use the log_value_vector_2.0.py to generate the csv file like Parameter_vector.png, which will be used to implement the anomaly detections later.

##  4. Model detection:
	Basiclly, we have two modules. 
	(1). Whereas, before implementing the modules, we will first see whether there is obvious malicious logs, we will report them first.
	(2). After that, we will first implement execution path anomaly detection with Execution_Path_Anomaly.py
	(3). Finally, we will implement parameter values anomaly detection with Parameter_value_performance_anomaly.py	



