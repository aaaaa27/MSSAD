from multiprocessing import Pool
import numpy as np
import time
import pandas as pd
import os
from numpy import array
import json
import re
from collections import OrderedDict

#merged_df = pd.merge(metric, trace, on='timestamp')

def merge_csvs(file_dir, save_path):
    file_list = os.listdir(file_dir)
    print(file_list)
    df = pd.read_csv(os.path.join(file_dir, file_list[0]))
    df.columns = [file_list[0].split('_')[0] + '_' + col if col != 'timestamp' else col for col in df.columns]
    for i in range(1, len(file_list)):
        df1 = pd.read_csv(os.path.join(file_dir, file_list[i]))
        df1.columns = [file_list[i].split('_')[0] + '_' + col if col != 'timestamp' else col for col in df1.columns]
        df = pd.merge(df, df1, on='timestamp')
    df.to_csv(os.path.join(save_path, 'merge_trace_metric.csv'), index=False)
    return df
trace_metric_dir = r'trace_metric'
merge_save_path = r'merge_trace_metric'

def generate_label(data_align, fault_inject):
    data_align['label'] = 0
    for i in range(fault_inject.shape[0]):
        fault_start = fault_inject.loc[i, 'startTime_s']
        fault_end = fault_inject.loc[i, 'endTime_s']
        data_align.loc[(data_align['timestamp'] >= fault_start) & (data_align['timestamp'] <= fault_end), 'label'] = 1
    data_align.to_csv(os.path.join(merge_save_path, 'merge_trace_metric_label.csv'), index=False)
    return data_align

def generate_label_rootcause(data_align, fault_inject,service_name_id):
    data_align['label'] = 0
    data_align['root_cause'] = None
    data_align['root_cause_id'] = None
    for i in range(fault_inject.shape[0]):
        service_name = fault_inject.loc[i, 'servicename']
        fault_start = fault_inject.loc[i, 'startTime_s']
        fault_end = fault_inject.loc[i, 'endTime_s']
        data_align.loc[(data_align['timestamp'] >= fault_start) & (data_align['timestamp'] <= fault_end), 'label'] = 1
        data_align.loc[(data_align['timestamp'] >= fault_start) & (data_align['timestamp'] <= fault_end), 'root_cause'] = service_name
        data_align.loc[(data_align['timestamp'] >= fault_start) & (data_align['timestamp'] <= fault_end), 'root_cause_id'] = service_name_id[service_name]

    data_align.to_csv(os.path.join(merge_save_path, 'merge_trace_metric_label_root_cause.csv'), index=False)
    return data_align
