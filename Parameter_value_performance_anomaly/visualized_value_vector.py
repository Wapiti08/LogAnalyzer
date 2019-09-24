import pandas as pd
import matplotlib.pyplot as plt


def visualize_value(filename):
    '''
    :param filename: the example is like '../data/System_logs/log_value_vector.csv'
    :return:
    '''
    fd = pd.read_csv(filename)
    parameter_value_vectors = []
    # get the parameter_value_vector line
    parameter_value_vectors = fd['parameter value vector']
    time_gap_lists = []
    # copy the orginal data used for analysis
    time_gap_lists = parameter_value_vectors.copy()
    time_gap_lists = [var.split(',')[0] for var in time_gap_lists]
    # transfer the str data into int dtype
    replace_pattern = { "[": "", "]": ""}
    # define the function to replace multiple values
    def replace_all(text, dic):
        for i, j in dic.items():
            text = text.replace(i, j)
        return text
    # replace the '[' and ']' in a string
    time_gap_lists = [int(replace_all(var, replace_pattern)) for var in time_gap_lists]
    plt.hist(time_gap_lists,bins=100)
    plt.xlabel('Time Gap')
    plt.ylabel('Occurrence')
    plt.title('Normal Linux Time Anomaly Detection')

if __name__ == '__main__':
    filename = '../data/System_logs/log_value_vector.csv'
    visualize_value(filename)