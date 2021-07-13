import pandas as pd
import numpy as np
from keras.models import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.utils import np_utils
from keras import Sequential
from keras.layers import Dense
import joblib
from pathlib import Path, PurePosixPath
from keras.callbacks import EarlyStopping
from pprint import pprint


def miss_rep_col(column_list):
    if column_list.dtype.name != 'category':
        # replace nan string to None
        return column_list.replace(np.nan, "None")
    else:
        # replace nan integer to 0
        return column_list.replace(np.nan, 0) 


def lab_enc(cate_list, label_encoder_file):
    # feature engineering for list
    cate_list_com = miss_rep_col(cate_list)
    
    if Path(label_encoder_file).is_file():
        label_encoder = joblib.load(label_encoder_file)
    else:
        laber_encoder = LabelEncoder()
        # key_log_arr = cate_list_com.values
        label_encoder = laber_encoder.fit(cate_list_com)
        # save the encoder for labelling
        joblib.dump(label_encoder, label_encoder_file.as_posix())
    
    label_encode_cate = label_encoder.transform(cate_list_com)

    return label_encode_cate


def encode_key(key_log_series, label_encoder_file, n_steps):
    ''' encode the string log key and one hot encode the number then

    '''
    label_encode_cate = lab_enc(key_log_series, label_encoder_file)
    # load label_encoder to return number of classes
    label_encoder = joblib.load(label_encoder_file)
    class_num = len(label_encoder.classes_)
    
    X, y = split_data(label_encode_cate, n_steps)

    # one hot encoding the labelled numeric log key array
    one_hot_y = np_utils.to_categorical(y)

    X = np.array(X).astype(float)
    
    return X , one_hot_y, class_num


def model_build(steps, class_num):
    ''' build the model with two hidden layers
    
    : return: compiled model
    '''
    model = Sequential()
    # input dim is the length of steps/history
    model.add(Dense(16, input_dim=steps, activation='relu'))
    model.add(Dense(16, activation='relu'))
    # output unit is the number of classes
    model.add(Dense(class_num,activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

    return model


def split_data(key_log_list, n_steps):
    X, y = [], []
    for i in range(len(key_log_list)):
        # create the end of position
        end_ix = i + n_steps
        # check whether the index excesses the boundary
        if end_ix > len(key_log_list) -1:
            break

        # get the input and output for model
        X_seq, Y_seq = key_log_list[i: end_ix], key_log_list[end_ix]
        # avoid arrays in a array
        X.append(X_seq.tolist())
        y.append(Y_seq.tolist())

    return X, y


def fit_eval(model, model_file, random_seed, X, y):
    
    if Path(model_file).is_file():
        model = load_model(model_file.as_posix())# fit the model
    else:
        print(X)
        earlystopping = EarlyStopping(monitor='accuracy', patience=10)
        

        # classifier = KerasClassifier(model, epochs=200, batch_size =5,  verbose=0)
        # evaluate the model
        # kfold = KFold(n_splits=10, shuffle=True, random_state=random_seed)
        print('=================')
        print(y)
        # results = cross_val_score(classifier, X, y, cv=kfold)
        # print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

        # in order to save model --- use fit
        model.fit(X, y, epochs=500, batch_size =5, validation_split =0.2, callbacks = [earlystopping], verbose=2)
    
        # saving weights
        model.save(model_file.as_posix())

    return model


def predict_sort(classifier, test_x, top_num):
    '''
    : param top_num: the top possibility to set as normal
    '''
    normal_result = []
    results = classifier.predict_proba(test_x, batch_size=16, verbose=1)
    # sort the result with shape of test_x.shape
    ## iterate the rows
    for row in range(results.shape[0]):
        # sort descending
        sorted_poss = [i for i in sorted(enumerate(results[row]), key=lambda x:x[1], reverse=True)]
        # extract the top num of possibilities tuple as the normal (class_num, possibility)
        normal_result.append(sorted_poss[: top_num])
    
    return normal_result


def rebuild_data(test_x, normal_result):
    ''' build normal sequences for updating 

    : param normal_result: checked predicted normal y -- list type with poss tuples inside
    '''
    assert test_x.shape[0] != len(normal_result), "Array Shape does not Match"
    # build normal sequence --- kind of N-gram
    normal_sequence = []
    
    for i in range(test_x.shape[0]):
        test_seq = test_x[i].tolist()
        normal_sequence.extend([ test_seq + [float(pro_y[0])]  for pro_y in normal_result[i]])

    return normal_sequence

    pprint(normal_sequence)


def anomaly_match(classifier, test_x, test_y, threshold):
    ''' match the top normal predict, anomaly logs if not matchable
    

    '''
    # pre_y is a list of predicted class number
    pre_y = classifier.predict_classes(test_x, batch_size=16, verbose=1)
    # save the possible prediction
    normal_result = []
    # save the seq and the prediction
    seq_pre_dict = {'seq_path':[], 'path_pred':[]}
    results = classifier.predict_proba(test_x, batch_size=16, verbose=1)
    # sort the result with shape of test_x.shape
    ## iterate the rows
    for row in range(results.shape[0]):
        # sort descending
        sorted_poss = [i for i in sorted(enumerate(results[row]), key=lambda x:x[1], reverse=True)]
        # pprint(sorted_poss)
        # print(sorted_poss)
        # extract the top num of possibilities tuple as the normal (class_num, possibility)
        pprint([clus_prob for clus_prob in sorted_poss if clus_prob[1] >= threshold])
        normal_result.append([clus_prob for clus_prob in sorted_poss if clus_prob[1] >= threshold])

    for i in range(len(pre_y)):
        # default add 0 as the prediction result to result column
        seq_pre_dict['seq_path'].append(list(test_x[i]) + [np.argmax(test_y[i])])
        # 0 is the normal
        seq_pre_dict['path_pred'].append(0)
        # print(np.argmax(test_y[i]))
        pprint([pre_y[0] for pre_y in normal_result[i]])
        if np.argmax(test_y[i]) not in [pre_y[0] for pre_y in normal_result[i]]:
            # change the prediction to anomaly with label 1
            seq_pre_dict['path_pred'][-1] = 1
            print("That is a potential anomaly prediction: {} from sequence {}".format(np.argmax(test_y[i]), test_x[i]))

    return seq_pre_dict    


def trace_seq_path(trace_df, seq_pre_dict):
    ''' generate dataframe to view the prediction

    : param trace_df: the dataframe with numeric log key, record_id inside        
    '''
    for key, value in seq_pre_dict.items():
        assert len(trace_df) == len(value)
        trace_df[key] = value

    return trace_df


def train_batch(exec_model, model_file, batch_x, batch_y, desired_proba, attempts):
    ''' update model with false positve and corrected wrong prediction
        stop train when predicted proba larger than a given value or attempts reach a given value

    : param exec_model: the original trained model
    : param batch_x: the x used to update the model
    : param batch_y: the normal or corrected y to update the prediction
    : param desired_proba: the threshold to stop the train_on_batch
    : param attempts: the threshold to stop the training
    
    : return: updated model with adjusted weights
    '''
    # train with batch data first
    exec_model.train_on_batch(batch_x, batch_y)
    # check the predict result
    pred_y = exec_model.predict_proba(batch_x, verbose=2)
    
    for i in range(len(batch_y)):
        # extract the predict proba for batch_y
        pre_proba = pred_y[i][int(np.argmax(batch_y[i]))]
        # set the exit condition
        success_flag = False
        no_of_attempts = 0
        # retrain on the single input and output
        while pre_proba <= desired_proba and (no_of_attempts<attempts):
            print(pre_proba)            
            exec_model.fit(np.reshape(batch_x[i],(1,-1)), np.reshape(batch_y[i],(1,-1)))
            
            no_of_attempts += 1

            pred_one_y = exec_model.predict_proba(np.reshape(batch_x[i],(1,-1)), verbose=2)
            pre_proba = pred_one_y[0][int(np.argmax(batch_y[i]))]
            
            print("Attempt Number %d, Predicted Proba for this iteration %f" %(no_of_attempts, pre_proba))

            if pre_proba > desired_proba:
                success_flag = True
                break

        if (success_flag == False) and (no_of_attempts >= attempts):
            print("[-] Failed to incorporate this feedback")

        if success_flag == True:
            print("[+] Feedback incorporated \n")
            print("Took %d iterations to learn!" %(no_of_attempts))

    # saving weights
    exec_model.save(model_file.as_posix())

    return exec_model


if __name__=="__main__":

    # read the data
    current_path = Path(__file__).resolve().parent
    df = pd.read_pickle(PurePosixPath(current_path/'log_value_vector.pkl'))
    encoder_file = PurePosixPath(current_path/"encoder.save")
    model_file = PurePosixPath(current_path/"model.h5")

    X, one_hot_y, class_num = encode_key(df['log key'], encoder_file, 5)
    model = model_build(5, class_num)
    classifier = fit_eval(model, model_file ,30, X, one_hot_y)
    # normal_result = predict_sort(classifier, X[:100], top_num=2)
    # pprint(normal_result)
    print('===========================')
    # pprint(rebuild_data(X[:100], normal_result))
    print('================================')
    # print(anomaly_match(classifier, X[:100], one_hot_y[:100], 2 ))
    seq_pre_dict = anomaly_match(classifier, X[:100], one_hot_y[:100], 0.1)
    trace_df = df.iloc[:100,]
    
    new_trace_df = trace_seq_path(trace_df, seq_pre_dict)
    
    print(new_trace_df)
    new_trace_df.to_csv(PurePosixPath(current_path/'trace_df.csv'))
    # train_batch(classifier, X[:100], one_hot_y[:100])
    # train_batch(classifier, X[:100], one_hot_y[:100], 0.5, 20)
