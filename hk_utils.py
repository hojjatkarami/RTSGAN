

def save_generated_data(generated_data, path2save, dataset="stock", seq_len=24):
    import numpy as np
    import pandas as pd
    from stock_energy.missingprocessor import Processor
    import pickle
    from fastNLP import DataSet

    data_path = "./TimeGAN/data"

    if 'stock' in dataset:
        loc = 'stock'
    elif 'energy' in dataset:
        loc = 'energy'
    # seq_len = 24
    df = pd.read_csv('{}/{}_data.csv'.format(data_path, loc), sep=",")
    types = ["continuous" for i in range(len(df.columns))]

    P = Processor(types)
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
