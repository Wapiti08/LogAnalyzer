'''
    standardization -- same position in the vector
    hstack -- stack columns 

'''
from sklearn.preprocessing import StandardScaler,RobustScaler, MinMaxScaler, Normalizer
from pathlib import Path, PurePosixPath
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.layers import Dense, LSTM, GRU
from keras import Sequential
from keras.callbacks import EarlyStopping
from numpy import subtract, square
import scipy.stats as st
import matplotlib.pyplot as plt
import keras

def miss_rep_col(col):
    ''' fill the missing data in parameter vector

    : param col: the single column in vector
    : return: pandas series column without missing data
    '''
    # check whether is categorical dtype
    if pd.Series(col).dtype == 'O':
        # replace nan string to None
        return pd.Series(col).replace(np.nan, "None")
    else:
        # replace nan integer to 0
        return pd.Series(col).replace(np.nan, 0) 


def lab_enc(cate_col, label_encoder_file):
    ''' encode categorical column in parameter vector to numeric data
    
    : cate_col: the single series column in vector
    : label_encoder_file: the path to save label encoder
    : return: encoded series numeric column 
    '''
    
    if Path(label_encoder_file).is_file():
        label_encoder = joblib.load(label_encoder_file)
    else:
        laber_encoder = LabelEncoder()
        # key_log_arr = cate_list_com.values
        label_encoder = laber_encoder.fit(cate_col)
        # save the encoder for labelling
        joblib.dump(label_encoder, label_encoder_file)

    label_encode_cate = label_encoder.transform(cate_col)

    return label_encode_cate


def stan_cols(col, col_ord, eventId, scaler_file):
    ''' normalize the matrix based on every column in parameter vector

    : param data: the parameter value matrix
    : param eventId: the event number in clusters
    : param scaler_file: the path to save/load scaler 
    : param col_ord: the order of col in a vector
    : return: normalized matrix
    '''
    
    scaler_path = Path(scaler_file).joinpath(str(eventId), str(col_ord) + 'scaler.save')
    
    if scaler_path.is_file():
        scaler = joblib.load(scaler_path.as_posix())
    
    else:
        scaler = StandardScaler() 
        # scaler = RobustScaler()
        # scaler = Normalizer()
        # reshape to 2D from 1D
        col = np.array(col).reshape(-1,1)
        scaler = scaler.fit(col)
    
    # standardilize column
    sat_col = scaler.transform(col)
    
    return sat_col


def split_data(data, n_steps):
    '''

    : param data: the matrix for one event cluster
    '''
    if isinstance(data, np.ndarray):
        length = data.shape[0]
    else:
        length = len(data)
    
    X, y = [], []
    for i in range(length):
        # create the end of position
        end_ix = i + n_steps
        # check whether the index excesses the boundary
        if end_ix > data.shape[0] -1:
            break

        # get the input and output for model
        X_seq, Y_seq = data[i: end_ix], data[end_ix]
        
        # avoid arrays in a array
        X.append(X_seq.tolist())
        y.append(Y_seq.tolist())

    return X, y

def model_build_train(train_X, train_y, model_file):
    '''
        the step is default one
    '''
    earlystopping = EarlyStopping(monitor='loss', patience=10)

    model = Sequential()
    model.add(LSTM(8,activation='relu',input_shape = (train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(LSTM(8,activation='relu'))
    model.add(Dense(train_y.shape[1]))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # fit model with validation
    model.fit(train_X, train_y, epochs=500, batch_size=16, callbacks = [earlystopping], validation_split=0.3, verbose=2, shuffle=False)
    
    # saving weights
    model.save(model_file.as_posix())

    return model


def mean_square_error(y_true, y_pred):
    ''' modified mse to compute squared error for parameter model evaluation

    :param y_true: the test y --- array
    :param y_pred: the predict y --- array
    :return: the mean of errors, the errors list
    '''
    # define the minus between two values
    # return original value
    d_matrix = subtract(y_true, y_pred)
    mses = []
    print('The shape of minus matrix is: {}'.format(d_matrix.shape))
    # compute mse for every row
    for i in range(d_matrix.shape[0]):
        # initialize to 0 for every new row
        sum_minus = 0
        for j in range(d_matrix.shape[1]):
            sum_minus += d_matrix[i, j] * d_matrix[i, j]
        # compute the mse for every row
        mse = np.mean(sum_minus)
        mses.append(mse)
    return mses


def model_predict(model, test_x, test_y):
    ''' 

    '''
    pre_y = model.predict(test_x, verbose=1)

    return mean_square_error(test_y, pre_y)


def confidence_interval(confidence, mse):
    ''' function to compute the confidence interval boundaries

    :param confidence: the confidence value or threshold, like 98%
    :param mses_list: the errors list
    :return: the boundaries
    '''
    # define the interval tuple
    return st.t.interval(confidence, len(mse)-1, loc=np.mean(mse), scale=st.sem(mse))


def anomaly_match(mses_list, fp_int, tp_int, eventId):
    '''
    : param mses_list: the list of mean square errors
    : param file_number: the matrix order
    : return: two thresholds (for false positive, true positive),
             the indexes of anomaly logs and false positive logs
    '''
    # here we use the max value as the threshold
    CI_fp1 = confidence_interval(fp_int, mses_list)
    
    # it is for the false positive detection
    threshold1 = CI_fp1[1]
    
    CI_an = confidence_interval(tp_int, mses_list)
    # save the result from prediction, index is the order in event matrix
    seq_pre_dict = {'seq_para':[],'para_pred':[]}
    # it is for the anomaly detection
    threshold2 = CI_an[1]

    print('[+] Reporting based on thresholds to match anomaly for Parameter Vector Model!')
    
    for i in range(len(mses_list)):
        seq_pre_dict['seq_para'].append(i)
        # default add 0 as normal
        seq_pre_dict['para_pred'].append(0)
        # compare the true positive predictions
        if mses_list[i] > threshold2:
            print('The {}th log in event {} sequence is potentially anomaly'.format(i, eventId))
            seq_pre_dict['para_pred'][-1] = 1
        # compare the false positive predictions
        elif mses_list[i] > threshold1:
            print('The {}th log in event {} sequence is false positive'.format(i, eventId))
            seq_pre_dict['para_pred'][-1] = 2
        else:
            continue

    return threshold1, threshold2, seq_pre_dict


def visual_mses(eventId, mses, threshold1, threshold2, CI1, CI2):
    ''' visualize the mse

    '''
    # create the x axis labels
    x_list = []
    for i in range(len(mses)):
        x_list.append(i)
    if len(x_list) < 1:
        return
    else:
        plt.plot(x_list, mses)
        # add the threshold lines with percentage
        plt.axhline(y=threshold1, color='b', linestyle="-", label='CI={}'.format(CI1))
        plt.axhline(y=threshold2, color='r', linestyle="-", label='CI={}'.format(CI2))
        plt.ylabel("Errors Values")
        # match the first num
        plt.title('Event '+ str(eventId) + ' ' + 'Errors Distribution')
        plt.legend()
        plt.show(block=False)
        plt.pause(3)
        plt.close()


def trace_seq_path(trace_df, seq_pre_dict, eventId, lab_encoder_file):
    ''' generate dataframe to view the prediction

    : param trace_df: the dataframe with numeric log key, record_id inside        
    '''
    # check whether the para_pred column has existed or not
    if 'para_pred' not in trace_df:
        # default assign 0
        trace_df['para_pred'] = 0

    lab_encoder = joblib.load(lab_encoder_file.as_posix())
    # {eventId: log message}

    event_log_map = dict(zip(lab_encoder.transform(lab_encoder.classes_), lab_encoder.classes_))
    # extract the original log message indexes
    ori_log = event_log_map[eventId]
    eventId_indexes = trace_df[trace_df['log key'] == ori_log].index
    
    assert len(eventId_indexes) != len(seq_pre_dict['para_pred']), " Length Not Matched "
    
    for i, index in enumerate(eventId_indexes):
        # replace the order with index
        seq_pre_dict['seq_para'][i] = int(index)
    
    for ord_index, df_index in enumerate(seq_pre_dict['seq_para']):
        trace_df['para_pred'][df_index] = seq_pre_dict['para_pred'][ord_index]

    return trace_df


def train_batch(para_model, model_file, batch_x, batch_y, steps, desired_thres, attempts):
    ''' update model with false positve and corrected wrong prediction
        stop train when predicted mes is smaller than a threshold or attempts reach a given num

    : param para_model: the original trained model
    : param batch_x: the x used to update the model
    : param batch_y: the normal prediction
    : param desired_thres: the threshold of confidence interval to match normal prediction
    : param attempts: the threshold to stop the training
    
    : return: updated model with adjusted weights
    '''
    # train with batch data first
    para_model.train_on_batch(batch_x, batch_y)
    # check the predict result
    mse_error = model_predict(para_model, batch_x, batch_y)
    # calculate the value matched the desired threshold for CI
    CI_AN = confidence_interval(desired_thres, mse_error)
    # compare every mse with the CI_AN
    for i in range(len(mse_error)):
        # set the exit condition
        success_flag = False
        no_of_attempts = 0
        # retrain if the mse is not acceptable
        while mse_error[i] > CI_AN[1] and (no_of_attempts < attempts):
            # convert 2D to 3D (samples, time steps, features)
            batch_x_one = np.reshape(batch_x[i], (1,batch_x[i].shape[0], batch_x[i].shape[1]))
            # convert 1D to 2D
            batch_y_one = np.reshape(batch_y[i], (1, len(batch_y[i])))
            para_model.fit(batch_x_one, batch_y_one)
            
            no_of_attempts += 1
            mse_one = model_predict(para_model, batch_x_one, batch_y_one)
            print("Attempt Number %d, Calculated error for this iteration %f" %(no_of_attempts, mse_one[0]))

            if mse_one < CI_AN[1]:
                success_flag = True
                break

        if (success_flag == False) and (no_of_attempts >= attempts):
            print("[-] Failed to incorporate this feedback")

        if success_flag == True:
            print("[+] Feedback incorporated \n")
            print("Took %d iterations to learn!" %(no_of_attempts))

    # saving weights
    para_model.save(model_file.as_posix())

    return para_model


if __name__ == "__main__":
    
    # load paths
    current_path = Path(__file__).resolve().parent
    df = pd.read_pickle(PurePosixPath(current_path/'Windows.log_structured.pkl'))
    model_file = PurePosixPath(current_path/"para_model.h5")
    scaler_file =  current_path
    label_file =  current_path
    trace_df = pd.read_csv(PurePosixPath(current_path/'trace_df.csv'))
    
    # load testing parameter value vector
    eventId = 16
    # with categorical data inside
    data = df[df['EventTemplate']=='A <*> <*> was <*>']['ParameterList']
    
    # feature engineering for parameter value matrix
    col_num = len(data[0])
    
    new_data = []

    # feature engineering for every single column
    for col_ord in range(col_num):
        
        new_data.append([row[col_ord] for row in data])
        # replace the missing values
        new_data[col_ord] = miss_rep_col(new_data[col_ord])

        # create paths to save encoder model
        label_encoder_path = Path(label_file).joinpath(str(eventId), str(col_ord) + 'label.save')
        
        if not Path(label_encoder_path).parent.is_dir():
            Path(label_encoder_path).parent.mkdir(parents=True, exist_ok=True)
        
        # encode categorical labels
        if pd.Series(new_data[col_ord]).dtype == 'O':
            new_data[col_ord] = lab_enc(new_data[col_ord], label_encoder_path.as_posix())            
        
        # nomalize the column
        # new_data[col_ord] = 
        # stan_cols(new_data[col_ord], col_ord, eventId, scaler_file)
        
        # reshape 2D to 1D
        new_data[col_ord] = np.reshape(new_data[col_ord],new_data[col_ord].shape[0])
    
    # shift the row to column   
    new_data = np.array(new_data).T
    n_steps = 5
    X, y = split_data(new_data, n_steps)
    
    # reshape x to (samples, time steps, features)
    train_X = np.array(X).reshape(-1, n_steps, len(data[0]))
    # reshape y to (samples, features)
    train_y = np.array(y).reshape(-1, len(data[0]))
    model = model_build_train(train_X, train_y, model_file)
    # model = keras.models.load_model(model_file)
    mse_error = model_predict(model, train_X[:50], train_y[:50])
    
    print(mse_error)
    # confidence = 0.99
    # print(confidence_interval(confidence, mse_error))
    fp_int = 0.97 
    tp_int = 0.999
    attempts = 10
    threshold1, threshold3, seq_pre_dict = anomaly_match(mse_error, fp_int, tp_int, eventId)
    # visual_mses(eventId, mse_error, threshold1, threshold3, fp_int, tp_int)
    lab_encoder_file = PurePosixPath(current_path/"encoder.save")
    trace_df = trace_seq_path(trace_df, seq_pre_dict, eventId, lab_encoder_file)
    trace_df.to_csv(PurePosixPath(current_path/'trace_df.csv'),index=False)
    steps = 5
    train_batch(model, model_file, train_X[:50], train_y[:50], steps, tp_int, attempts)
   
