import sys
sys.path.append('../')

import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize, MinMaxScaler
import re
from math import sqrt, pow
import os
import joblib
from keras.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro, normaltest

# load the visualized part (if needed)
from Parameter_value_performance_anomaly import visualized_value_vector
# load the module to transform the value vector to numerical data --- step1
from Parameter_value_performance_anomaly import transform_numerical_data
# load the module to generate the matrix for every log key sequence --- step2
from Parameter_value_performance_anomaly import gen_log_key_matrix
# load the module to analyse the matrix for anomaly detection --- step3
from Parameter_value_performance_anomaly import matrix_analyse_report_anomaly
import optparse
# compute the confidence intervial
from Parameter_value_performance_anomaly import anomaly_predict


if __name__ == "__main__":

    # set the format of command input
    parser = optparse.OptionParser('usage %prog --p1 <log_value_vector> --p2 <key_num_para_dict> --p3 <Event_npy>')
    # set the parameter
    parser.add_option('--p1', dest = 'log_value_vector_filename', type = 'string', help = 'Please input the path of the log_value_vector file:')
    parser.add_option('--p2', dest = 'para_dict_filename', type = 'string', help = 'Please input the path of the key_num_para_dict file:')
    parser.add_option('--p3', dest = 'Event_npy_folder', type = 'string', help = 'Please input the folder to save the event matrix for every log key:')

    # parser arguments through the parse_args()
    (options, args) = parser.parse_args()
    log_value_vector_filename = options.log_value_vector_filename
    para_dict_filename = options.para_dict_filename
    Event_npy_folder = options.Event_npy_folder

# ================== part1 to load the dataframe for parameter detection =================

    log_value_vector_csv_fd = pd.read_csv(log_value_vector_filename)

# ================== part2 to transform the textual value to numerical data ================

    key_para_dict_filename = '../Dataset/Linux/Client/Client_structured/key_para_dict.csv'
    key_para_dict, fd_id = transform_numerical_data.vocabulary_generate(log_value_vector_csv_fd, key_para_dict_filename)

    # module to process the exception in template computation
    tokens = transform_numerical_data.tokens_generate(key_para_dict)
    # save the tokens_dict_filename
    tokens_dict_filename = '../Dataset/Linux/Client/Client_structured/tokens_dict.pkl'
    tokens_encode_dict = transform_numerical_data.token_dict(tokens, tokens_dict_filename)

    # split the parameter value vector into different columns
    # fd_id is the copied csv and list_name is like value x
    fd_id, list_name = transform_numerical_data.split_vectors(fd_id, log_value_vector_filename)

    # replace the textual data to numerical data
    fd_value = transform_numerical_data.map_vectors(fd_id, list_name, log_value_vector_filename, tokens_encode_dict)

    # integrate the vector lines into one
    integrated_fd_value = transform_numerical_data.integrate_lines(fd_value, list_name)

    # delete repeated column in the csv
    transform_numerical_data.delete_repeated_line(integrated_fd_value, log_value_vector_filename)


# ================== part3 to generate the separate matrix for log key ===================

    # we have the parameter file --- log_value_vector_filename
    fd_parameter = pd.read_csv(log_value_vector_filename)
    fd_parameter = fd_parameter.copy()
    # create the aim file where the key_num_para_dict.csv will be saved
    key_num_para_dict = gen_log_key_matrix.log_vectors(fd_parameter, para_dict_filename)

    # create all the matrixes for all the eventIDs
    for key in key_num_para_dict.keys():
        print("the key is:", key)
        gen_log_key_matrix.str_array(key_num_para_dict, key, Event_npy_folder)

# ================== part4 to analyse the matrix ===================

    filenames = []
    root_dir = '../Dataset/Linux/Client/Client_structured/Event_npy/'
    # r=root, d = directories, f=files
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)
    else:
        for r, d, f in os.walk(root_dir):
            for file in f:
                if file.endswith('.npy'):
                    filenames.append(os.path.join(r, file))
    # set the random seed
    seed = 7
    rmses = []
    rmses_dict = {}

    # record the anomaly logs with the name of file and the anomaly logs order
    suspicious_anomaly_dict, fp_logs_dict = {}, {}

    # identify whether result file has been generated before
    for file in filenames:
        if os.path.isfile(file + '_rmses.pkl'):
            rmses = joblib.load(file + '_rmses.pkl')

        else:
            # looping read single file
            print("we are processing matrix:", file)
            matrix = np.load(file)
            # set n_steps_in and n_steps_out depending on the sequence length of matrix
            # we set the test_size=0.4, the length of matrix should be at least 8
            # Here, I will change the length of history to see the performance
            if matrix.shape[0] >= 8:
                n_steps = 3
                X, Y = matrix_analyse_report_anomaly.training_data_generate(matrix, n_steps)
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.4, random_state=seed)

            elif matrix.shape[0] >= 4:
                n_steps = 1
                X, Y = matrix_analyse_report_anomaly.training_data_generate(matrix, n_steps)
                # test_x and
                train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.5, random_state=seed)
            else:
                continue
            # get the model
            model = matrix_analyse_report_anomaly.LSTM_model(train_x, train_y)
            print("the test_x is:", test_x)
            # make a prediction
            yhat = model.predict(test_x)
            # delete the time step element
            print("the predicted y is:", yhat)

            rmse, means = matrix_analyse_report_anomaly.mean_squared_error_modified(test_y, yhat)
            # rmse, meams = mean_squared_error(test_y, yhat)
            rmse = sqrt(rmse)
            print('Test RMSE: %.3f' % rmse)
            # use the mean square error to compare the difference between predicted y and validation y
            # the error follows the Gaussian distribution ---- normal, otherwise abnormal
            rmses.append(rmse)
            # save the result
            rmses_dict[file] = rmse
            # save the results to files
            joblib.dump(rmses, file + '_rmses.pkl')
            joblib.dump(rmses_dict, file + '_rmses_dict.pkl')

        # ===== part to predict the anomaly logs ====


        file_number = re.findall('\d+', file)
        threshold1, threshold2, threshold3, suspicious_logs, \
            fp_logs = anomaly_predict.anomaly_report(rmses, file_number)

        # part to print the picture of means with bar chart
        # create the x axis labels for plot
        x_list = []
        for i in range(len(rmses)):
            x_list.append(i)
        # check the length of rmses
        if len(x_list) <= 1:
            pass
        else:
            # part to print the picture of means with line chart
            plt.plot(x_list, rmses)
            # add the threshold lines with percentage
            plt.axhline(y=threshold1, linestyle = "-", label = '98%')
            plt.axhline(y=threshold2, linestyle = "-", label = '99%')
            plt.axhline(y=threshold3, linestyle = "-", label = '99.9%')
            plt.ylabel("Errors Values")

            plt.title(file_number[0] + ' ' + 'Errors Distribution')
            # plt.title(file + ' ' + 'Errors Distribution')
            plt.show()
        # generate the dict about anomaly and false positive logs
        suspicious_anomaly_dict[file_number[0]] = suspicious_logs
        fp_logs_dict[file_number[0]] = fp_logs
    # save the result
    joblib.dump(suspicious_anomaly_dict,'./result/suspicious_anomaly.pkl')
    joblib.dump(fp_logs_dict, './result/fp_logs.pkl')

