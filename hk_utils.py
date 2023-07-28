

def save_generated_data(generated_data, path2save, dataset="stock", seq_len=24):
    import numpy as np
    import pandas as pd
    from stock_energy.missingprocessor import Processor
    import pickle
    from fastNLP import DataSet

    if 'stock' in dataset:
        data_path = "./TimeGAN/data"
        loc = 'stock'
        df = pd.read_csv('{}/{}_data.csv'.format(data_path, loc), sep=",")
        types = ["continuous" for i in range(len(df.columns))]
        P = Processor(types)

    elif 'energy' in dataset:
        data_path = "./TimeGAN/data"
        loc = 'energy'
        df = pd.read_csv('{}/{}_data.csv'.format(data_path, loc), sep=",")
        types = ["continuous" for i in range(len(df.columns))]
        P = Processor(types)

    elif 'sine' in dataset:
        data_path = "../COSCI-GAN/Dataset/"
        loc = "data_frame_sine_normal.pkl"
        seq_len = 800
        with open(data_path+loc, "rb") as f:
            # [n_samples, seq_len * n_features]
            data = pickle.load(f).iloc[:, :-1].values
        data2 = np.stack([data[:, :seq_len].flatten(),
                          data[:, seq_len:].flatten()], axis=1)
        df = pd.DataFrame(data2)
        types = ["continuous" for i in range(len(df.columns))]  # [n_feature]
        P = Processor(types)

    # seq_len = 24

    # Flip the data to make chronological data
    # ori_data = P.fit_transform(df)
    # ori_data = ori_data[::-1]

    # temp_data = [ori_data[i:i + seq_len]
    #              for i in range(0, len(ori_data) - seq_len)]

    temp_data = [np.array(sample) for sample in generated_data]
    dataset = DataSet(
        {"seq_len": [seq_len] * len(temp_data), "dyn": temp_data, "sta": [0]*len(temp_data)})
    syn_dataset = {
        "gen_set": dataset,
        "dynamic_processor": P,
        "static_processor": Processor([])
    }
    from utils.general import make_sure_path_exists
    make_sure_path_exists("./data")
    with open(path2save+'/syn_dataset.pkl', "wb") as f:
        pickle.dump(syn_dataset, f)

    return syn_dataset
