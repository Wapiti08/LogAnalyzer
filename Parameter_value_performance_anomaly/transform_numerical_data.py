'''
This part is to transform all the textual context in parameter value
vector line to numerical data, in order to make that parameters suitable
to analyse for LSTM network
'''

import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from pandas import Series


# module to process the exception in template computation
def template_filter(parameters):
    # pattern to filter
    pattern = 'tty=|rhost=|user='
    for parameter in parameters:
        parameter = re.sub(pattern, '', parameter)
    return parameters


def key_to_EventId(df):

    '''
    :param df: normaly, the log key column in df is hashed values
    :return: log_key_sequence: the column of log key
             key_name_dict: format is {Exx: SRWEDFFW(hashed value),...}
             K: the number of unique log key events
    '''

    df = df.copy()
    log_key_sequence = df['log key']
    log_key_sequence = list(log_key_sequence)
    # get the unique list
    items = set(log_key_sequence)
    # define the total number of log keys
    K = None
    K = len(items)
    print("the length of log_key_sequence is:", len(items))
    key_name_dict = {}

    for i, item in enumerate(items):
        # items is a set
        # columns are the lines of log key sequence
        for j in range(len(log_key_sequence)):
            if log_key_sequence[j] == item:
                name = 'E' + str(i)
                # log_key_sequence[j]='k'+str(i)
                key_name_dict[name] = log_key_sequence[j].strip('\n')

    return log_key_sequence, key_name_dict, K


# ================= get the vocabulary set ==================
def vocalubary_generate(fd):
    '''
    :param fd:  pandas dataframe with the log key column in it is hashed values
    :return: fd_id: copied fd dataframe, in order to protect the original data
             key_para_dict: the format is {Exx:[textual parameter 1],[textual parameter 2],...}
    '''

    key_para_dict = {}

    log_key_sequence, key_name_dict, K = key_to_EventId(fd)
    fd_id = fd.copy()
    # swith the key and value in a dict
    key_name_dict_rev = dict((value,key) for key,value in key_name_dict.items())
    # mapping the value to keyID
    fd_id['log key'] = fd_id['log key'].map(key_name_dict_rev)

    uni_log_key_id = list(set(fd_id['log key']))

    parameters = []

    for i in range(len(uni_log_key_id)):
        # get all the parameters with the same eventID
        parameters = fd_id[fd_id['log key'] == uni_log_key_id[i]]['parameter value vector']
        parameters = template_filter(parameters)
        key_para_dict[uni_log_key_id[i]] = parameters.values[:]

    # padding nan to object without enough length
    df_dict_para = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict.items()]))
    df_dict_para.to_csv('../Dataset/Linux/Malicious_Separate_Structured_Logs/key_para_dict.csv',index= False, header=key_para_dict.keys())
    return key_para_dict, fd_id



def tokens_generate(key_para_dict):
    '''
    :param key_para_dict: the format is {Exx:[textual parameter 1],[texual parameter 2],...}
    :return: tokens: all the word tokens in the parameter value vector column
    '''
    text = []
    for key, value in key_para_dict.items():
        # extract the time part from values
        for i in range(len(value[:])):
            if value[i].split(',')[1:] == []:
                break
            else:
                value[i] = re.sub('[\[|\]|\'|\|\s+|\.|\-]', '', str(value[i])).split(',')
                if value[i] == ['']:
                    break
                else:
                    text.append([var for var in value[i][1:]])

    # get the text for token_nize
    tokens = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            tokens.append(text[i][j])
    # delete the blank value
    tokens = [var for var in tokens if var]
    tokens = set(tokens)

    return tokens

def token_dict(tokens):
    '''
    :param tokens: all the word tokens in the parameter value vector column
    :return: token_encode_dict: the format is ['fawjeiajet';[32,45,65,..],...]
    '''

    # build the dict about different value
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    encoded_texts = tokenizer.texts_to_sequences(tokens)
    # build the dict with tokens --> encoded_texts
    token_encode_dict = {}
    for token, encoded_text in zip(tokens, encoded_texts):
        token_encode_dict[token] = encoded_text

    return token_encode_dict


def split_vectors(fd_id, filename):
    '''
    :param fd_id: copied fd dataframe, in order to protect the original data
    :return: fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
            list_name: the format is: value0, value1, value2, ....
    '''
    list_length = []
    for var in fd_id['parameter value vector']:
        list_length.append(len(var.split(',')))
    # max(list_length) ---- 16
    # list_length
    list_name = []
    for i in range(max(list_length)):
        list_name.append('value' + str(i))
    fd_id[list_name] = fd_id['parameter value vector'].str.split(",", expand=True, )
    # [var for var in fd_id['value15'] if var]
    # fd_id
    for name in list_name:
        for var in range(len(fd_id[name])):
            # we should use fd_id[x] to rewrite value in
            if fd_id[name][var] != None:
                fd_id[name][var] = re.sub("[\[|\]|']|\s+|\.|\-", '', fd_id[name][var])
    fd_id.to_csv(filename, index=False)

    return fd_id, list_name


def map_vectors(fd_id, list_name, filename):
    '''
    :param fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
    :param list_name: the format is: value0, value1, value2, ....
    :return: fd_value: csv with textual values in parameter value vector replaced by numerical values
    '''
    # fd_value = pd.read_csv('../data/System_logs/log_value_vector_value.csv',delimiter=',', skipinitialspace=True)
    fd_value = fd_id
    for var in range(1,len(list_name)):
        fd_value[list_name[var]] = fd_value[list_name[var]].map(token_encode_dict)
    fd_value.to_csv(filename, index = False)

    return fd_value


def integrate_lines(fd_value):
    fd_value['ColumnX'] = fd_value[fd_value.columns[3:19]].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
    fd_value = fd_value.drop(['parameter value vector'], axis=1)
    fd_value = fd_value.drop(list_name, axis=1)

    return fd_value

def delete_repeated_line(fd_value,filename):
    fd_value['parameter value vector'] = fd_value['ColumnX']
    fd_value = fd_value.drop(['ColumnX'], axis=1)
    fd_value.to_csv(filename, index=False)


if __name__ == '__main__':
    # get the log_value_vector csv (log message, log key, parameter value vector)
    log_value_vector_csv = '../Dataset/Linux/Malicious_Separate_Structured_Logs/log_value_vector_mali.csv'
    fd = pd.read_csv(log_value_vector_csv)

    # input the normal fd file
    key_para_dict, fd_id = vocalubary_generate(fd)
    # input the abnormal fd file

    # module to process the exception in template computation



    tokens = tokens_generate(key_para_dict)
    # the format is 'ate awte awet':[34,234,13]
    token_encode_dict = token_dict(tokens)

    # split the parameter value vector into different columns
    fd_id , list_name = split_vectors(fd_id, log_value_vector_csv)

    # replace the textual data to numerical data
    fd_value = map_vectors(fd_id, list_name, log_value_vector_csv)
    # integrate the vector lines into one
    integrated_fd_value = integrate_lines(fd_value)
    # delete repeated column in csv
    delete_repeated_line(integrated_fd_value, log_value_vector_csv)