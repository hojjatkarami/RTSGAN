from utils.general import make_sure_path_exists
from fastNLP import DataSet
import numpy as np
import pandas as pd
from stock_energy.missingprocessor import Processor
import pickle

data_path = "./TimeGAN/data"
loc = "stock"
seq_len = 24
df = pd.read_csv('{}/{}_data.csv'.format(data_path, loc), sep=",")
types = ["continuous" for i in range(len(df.columns))]

P = Processor(types)
# Flip the data to make chronological data
ori_data = P.fit_transform(df)
ori_data = ori_data[::-1]

temp_data = [ori_data[i:i + seq_len]
             for i in range(0, len(ori_data) - seq_len)]

dataset = DataSet({"seq_len": [seq_len] * len(temp_data),
                  "dyn": temp_data, "sta": [0]*len(temp_data)})
dic = {
    "train_set": dataset,
    "dynamic_processor": P,
    "static_processor": Processor([])
}
print(P.dim, len(temp_data))
make_sure_path_exists("./data")
with open("./data/{}.pkl".format(loc), "wb") as f:
    pickle.dump(dic, f)


data_path = "./TimeGAN/data"
loc = "energy"
seq_len = 24
df = pd.read_csv('{}/{}_data.csv'.format(data_path, loc), sep=",")
types = ["continuous" for i in range(len(df.columns))]

P = Processor(types)
# Flip the data to make chronological data
ori_data = P.fit_transform(df)
ori_data = ori_data[::-1]

temp_data = [ori_data[i:i + seq_len]
             for i in range(0, len(ori_data) - seq_len)]

dataset = DataSet({"seq_len": [seq_len] * len(temp_data),
                  "dyn": temp_data, "sta": [0]*len(temp_data)})
dic = {
    "train_set": dataset,
    "dynamic_processor": P,
    "static_processor": Processor([])
}
print(P.dim, len(temp_data))
make_sure_path_exists("./data")
with open("./data/{}.pkl".format(loc), "wb") as f:
    pickle.dump(dic, f)


# prepare data_frame_sine_normal

data_path = "../COSCI-GAN/Dataset/"
loc = "data_frame_sine_normal.pkl"
seq_len = 800


with open(data_path+loc, "rb") as f:
    # [n_samples, seq_len * n_features]
    data = pickle.load(f).iloc[:, :-1].values
# [n_samples* seq_len , n_features]
data2 = np.stack([data[:, :seq_len].flatten(),
                 data[:, seq_len:].flatten()], axis=1)

# DEBUG
# print(data.shape)
# print(data[0,:20])
# print(data[0,800:820])

# data2.shape
# data2[:20,:]


df = pd.DataFrame(data2)
types = ["continuous" for i in range(len(df.columns))]  # [n_feature]

# Flip the data to make chronological data
ori_data = P.fit_transform(df)
# ori_data = ori_data[::-1]
# print(ori_data[:5,:])
# temp_data = [ori_data[i:i + seq_len] for i in range(0, len(ori_data) - seq_len)]


# [(n_data-n_seq) * [n_seq * n_feature]]]
temp_data = [ori_data[i:i + seq_len] for i in range(0, len(ori_data), seq_len)]

# DEBUG
# a = [i for i in range(0, len(ori_data),seq_len)]
# len(ori_data), a[:5], a[-5:]
# len(temp_data), temp_data[0].shape
# temp_data[0][:5]


dataset = DataSet({"seq_len": [seq_len] * len(temp_data),
                  "dyn": temp_data, "sta": [0]*len(temp_data)})
dic = {
    "train_set": dataset,
    "dynamic_processor": P,
    "static_processor": Processor([])
}
print(P.dim, len(temp_data))
make_sure_path_exists("./data")
with open("./data/{}".format(loc), "wb") as f:
    pickle.dump(dic, f)
