from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def getTimeEmbedding(time):
    """
    这个函数把时间embedding了，将每个时间转换位月，天，周，小时，分。 范围是（0.5——-0.5）
    """
    df = pd.DataFrame(time, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])

    df['minute'] = df['time'].apply(lambda row: row.minute / 59 - 0.5)
    df['hour'] = df['time'].apply(lambda row: row.hour / 23 - 0.5)
    df['weekday'] = df['time'].apply(lambda row: row.weekday() / 6 - 0.5)
    df['day'] = df['time'].apply(lambda row: row.day / 30 - 0.5)
    df['month'] = df['time'].apply(lambda row: row.month / 365 - 0.5)

    return df[['minute', 'hour', 'weekday', 'day', 'month']].values

def getStable(data, w=1440):
    # 转换为 DataFrame 并计算趋势
    trend = pd.DataFrame(data).rolling(w, center=True).median().values
    stable = data - trend

    # 去掉滚动窗口带来的边界影响
    stable_trimmed = stable[w // 2:-w // 2, :]
    trend_trimmed = trend[w // 2:-w // 2, :]

    return stable_trimmed, trend_trimmed

class D3R_dataset(Dataset):
    """
    构建dataset
    """
    def __init__(self, data, time, stable, label, window_size):
        self.data = data
        self.time = time
        self.stable = stable
        self.label = label
        self.window_size = window_size

    def __getitem__(self, index):
        data = self.data[index: index + self.window_size, :]
        time = self.time[index: index + self.window_size, :]
        stable = self.stable[index: index + self.window_size, :]
        label = self.label[index: index + self.window_size, :]

        return data, time, stable, label

    def __len__(self):
        return len(self.data) - self.window_size + 1

def get_dataset(config):
    period=config['period']
    train_rate=config['train_rate']
    init_data=np.load(config['train_data_path'])
    init_time=getTimeEmbedding(np.load(config['train_date_path']))

    test_data=np.load(config['test_data_path'])
    test_time=getTimeEmbedding(np.load(config['test_date_path']))
    test_label=np.load(config['test_label_path'])

    scaler=StandardScaler()
    scaler.fit(init_data)

    init_data = pd.DataFrame(scaler.transform(init_data)).fillna(0).values
    test_data = pd.DataFrame(scaler.transform(test_data)).fillna(0).values

    init_data, init_stable = getStable(init_data, w=period)
    init_time = init_time[period // 2:-period // 2, :]
    init_label = np.zeros((len(init_data), 1))
    test_stable = np.zeros_like(test_data)

    train_data = init_data[:int(train_rate * len(init_data)), :]
    train_time = init_time[:int(train_rate * len(init_time)), :]
    train_stable = init_stable[:int(train_rate * len(init_stable)), :]
    train_label = init_label[:int(train_rate * len(init_label)), :]

    valid_data = init_data[int(train_rate * len(init_data)):, :]
    valid_time = init_time[int(train_rate * len(init_time)):, :]
    valid_stable = init_stable[int(train_rate * len(init_stable)):, :]
    valid_label = init_label[int(train_rate * len(init_label)):, :]

    config['feature_num']=train_data.shape[1]
    config['time_num']=train_time.shape[1]

    train_dataset=D3R_dataset(train_data, train_time, train_stable, train_label, config['window_size'])
    valid_dataset=D3R_dataset(valid_data, valid_time, valid_stable, valid_label, config['window_size'])
    test_dataset=D3R_dataset(test_data, test_time, test_stable, test_label, config['window_size'])
    init_dataset=D3R_dataset(init_data, init_time, init_stable, init_label, config['window_size'])
    return train_dataset, valid_dataset, test_dataset, init_dataset
