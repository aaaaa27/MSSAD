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

#读取文件夹下所有csv文件，按timestamp进行拼接，存储到新的csv文件中
def merge_csvs(file_dir, save_path):
    file_list = os.listdir(file_dir)
    print(file_list)
    df = pd.read_csv(os.path.join(file_dir, file_list[0]))
    df.columns = [file_list[0].split('_')[0] + '_' + col if col != 'timestamp' else col for col in df.columns]
    for i in range(1, len(file_list)):
        df1 = pd.read_csv(os.path.join(file_dir, file_list[i]))
        df1.columns = [file_list[i].split('_')[0] + '_' + col if col != 'timestamp' else col for col in df1.columns]
        #按timestamp对df和df1进行merge
        df = pd.merge(df, df1, on='timestamp')
    df.to_csv(os.path.join(save_path, 'merge_trace_metric.csv'), index=False)
    return df
trace_metric_dir = r'E:\MS\data\eadro-main\eadro-main\data\SN_process\data\trace_metric'
merge_save_path = r'E:\MS\data\eadro-main\eadro-main\data\SN_process\data\merge_trace_metric'
#将所有微服务的trace_metric数据进行merge
#merge_csvs(trace_metric_dir, merge_save_path)



#按照故障注入数据，为merge_trace_metric生成label列
def generate_label(data_align, fault_inject):
    data_align['label'] = 0
    for i in range(fault_inject.shape[0]):
        fault_start = fault_inject.loc[i, 'startTime_s']
        fault_end = fault_inject.loc[i, 'endTime_s']
        data_align.loc[(data_align['timestamp'] >= fault_start) & (data_align['timestamp'] <= fault_end), 'label'] = 1
    data_align.to_csv(os.path.join(merge_save_path, 'merge_trace_metric_label.csv'), index=False)
    return data_align

fault_inject = pd.read_csv(r'E:\MS\data\eadro-main\eadro-main\data\SN\SN_fault_data.csv')
T_M_data_align = pd.read_csv(os.path.join(merge_save_path, 'merge_trace_metric.csv'))
#generate_label(T_M_data_align, fault_inject)


#按照故障注入数据，为merge_trace_metric生成label列和root_cause列
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

service_name = ["compose-post-service", "home-timeline-service", "media-service",
                           "nginx-web-server", "post-storage-service", "social-graph-service",
                           "text-service", "unique-id-service", "url-shorten-service", "user-mention-service",
                           "user-service","user-timeline-service"]
#servicename转为id
service_name_id = {service_name[i]: i for i in range(len(service_name))}

fault_inject_r = pd.read_csv(r'E:\MS\data\eadro-main\eadro-main\data\SN\SN_fault_data.csv')
T_M_data_align_r = pd.read_csv(os.path.join(merge_save_path, 'merge_trace_metric.csv'))
generate_label_rootcause(T_M_data_align_r, fault_inject_r,service_name_id)