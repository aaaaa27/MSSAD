#将所有的fault数据提取出来，存储故障数据到为csv文件fault_csv_path中
import pandas as pd
import json
import os
import pytz
import time
from datetime import datetime
tz = pytz.timezone('Asia/Shanghai')

def ts_to_date(timestamp):
    # 将秒级别的时间戳转换为秒级别的
    #timestamp = timestamp // 1000000
    #将秒级别的时间戳转换为时间
    try:
        return datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S.%f')
    except:
        return datetime.fromtimestamp(timestamp, tz).strftime('%Y-%m-%d %H:%M:%S')


def time_to_ts(ctime):
    try:
        timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S.%f')
    except:
        try:
            timeArray = time.strptime(ctime, '%Y-%m-%d %H:%M:%S')
        except:
            timeArray = time.strptime(ctime, '%Y-%m-%d')
    return int(time.mktime(timeArray)) * 1000

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
#读取csv文件到DataFrame
def read_csv(file_path):
    data = pd.read_csv(file_path)
    return data
#将fault数据转换为csv文件
def fault_to_csv(fault_data, fault_csv_path):
    fault_data_csvs = pd.DataFrame(
        columns=['datetime', 'servicename', 'fault', 'startTime_s', 'duration','endTime_s'])
    for faults in fault_data:
        fault_data_list = faults['faults']
        for fault in fault_data_list:
            fault_data_csvs = fault_data_csvs._append(
                {'datetime': ts_to_date(fault['start']),
                 #将name字段按照_分割，取第一个和最后一个元素，返回string
                 #'servicename': ' '.join(fault['name'].split('_')[1:-1]),  #TT数据集中的name字段
                 'servicename': fault['name'],  #SN数据集中的name字段
                 'fault': fault['fault'],
                 'startTime_s': fault['start'],
                 'duration': fault['duration'],
                 'endTime_s': fault['start']+fault['duration'],},
                ignore_index=True)
    fault_data_csvs.to_csv(fault_csv_path, index=True)
    return fault_data_csvs
span_data_dir = r'E:\MS\data\eadro-main\eadro-main\data'
#dataset = ['SN','TT']
dataset = ['SN']
for ds in dataset:
    fault_data = []
    for path in os.listdir(os.path.join(span_data_dir, ds)):
        #判断path的结尾是否为.json并且开头是否为SN.fault
        #if path.startswith('TT.fault') and path.endswith('.json'): #判断TT数据集中的fault数据
        if path.startswith('SN.fault') and path.endswith('.json'): #判断SN数据集中的fault数据
            fault_json_path = os.path.join(span_data_dir, ds, path)
            row_fault_data = [read_json(fault_json_path)]
            fault_data = fault_data + row_fault_data   #拼接所有的fault数据
    fault_csv_path = os.path.join(span_data_dir, ds, f'{ds}_fault_data.csv')
    all_fault_data = fault_to_csv(fault_data, fault_csv_path)



