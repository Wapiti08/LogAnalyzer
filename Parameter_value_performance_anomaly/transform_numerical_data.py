'''
This part is to transform all the textual context in parameter value
vector line to numerical data, in order to make that parameters suitable
to analyse for LSTM network
'''

import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from pandas import Series
import joblib

# module to process the exception in template computation
def template_filter(parameters):
    # pattern to filter
    pattern = 'tty=|rhost=|user='
    for parameter in parameters:
        parameter = re.sub(pattern, '', parameter)
    return parameters


# ================= get the vocabulary set ==================
def vocabulary_generate(fd, key_para_dict_filename, key_para_dict_index_filename):
    '''
    :param fd:  pandas dataframe with the log key column in it is hashed values
    :return: fd_id: copied fd dataframe, in order to protect the original data
             key_para_dict: the format is {Exx:[textual parameter 1],[textual parameter 2],...}
    '''

    # key_para_dict will save the some key name Exx with all parameters
    key_para_dict, key_para_dict_index = {}, {}
    log_key_sequence, key_name_dict, K = key_to_EventId(fd)
    fd_id = fd.copy()
    # switch the key and value in a dict
    key_name_dict_rev = dict((value,key) for key,value in key_name_dict.items())
    # mapping the value to keyID
    fd_id['log key'] = fd_id['log key'].map(key_name_dict_rev)

    uni_log_key_id = list(set(fd_id['log key']))

    parameters, parameters_index = [], []

    for i in range(len(uni_log_key_id)):
        # get all the parameters with the same eventID
        parameters = fd_id[fd_id['log key'] == uni_log_key_id[i]]['parameter value vector']
        print(parameters)
        para_index = parameters.index.values
        # filter special characters
        parameters = template_filter(parameters)
        # transform the series object to list type
        parameters_index = list(parameters)
        parameters_index.insert(0, str(para_index))
        key_para_dict[uni_log_key_id[i]] = parameters.values[:]
        key_para_dict_index[uni_log_key_id[i]] = parameters_index

    # padding nan to object without enough length
    df_dict_para = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict.items()]))
    df_dict_para_index = pd.DataFrame(dict([(k,Series(v)) for k,v in key_para_dict_index.items()]))

    df_dict_para.to_csv(key_para_dict_filename,index = False, header=key_para_dict.keys())
    # add the index of original index to the dict
    df_dict_para_index.to_csv(key_para_dict_index_filename, index = False, header=key_para_dict_index.keys())
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


def token_dict(tokens, tokens_dict_filename):
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
    joblib.dump(token_encode_dict, tokens_dict_filename)

    return token_encode_dict

# split one column into different columns according to cluster name
def split_vectors(fd_id, filename):
    '''
    :param fd_id: copied fd dataframe, in order to protect the original data
    :param filename: the location to store csv
    :return: fd_id: csv with parameter value vector splitted into various columns according to the max length of vector
            list_name: the format is: value0, value1, value2, ....
    '''
    list_length = []
    for var in fd_id['parameter value vector']:
        list_length.append(len(var.split(',')))

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
                fd_id[name][var] = fd_id[name][var].str.replace("[\[|\]|']|\s+|\.|\-", '')
    fd_id.to_csv(filename, index=False)

    return fd_id, list_name


def map_vectors(fd_id, list_name, filename, token_encode_dict):
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

def integrate_lines(fd_value, filename):
    ''''''
    # integrate multiple parameter columns into one column
    df_num_para = fd_value.copy()
    df_num_para['parameter value vector'] = df_num_para[df_num_para.columns[3:]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    # drop the splitted columns
    df_num_para = df_num_para.drop(df_num_para.columns[3:], axis=1)

    # save the integrated result
    df_num_para.to_csv(df_num_para_filename, index=False)

    return df_num_para
