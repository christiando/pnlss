import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.join(os.getcwd(),'repos/darts/'))
from darts import TimeSeries

import jax
from jax import config

device_kwargs = {"accelerator": "gpu"}


# load vanderpol data
from darts.models import NBEATSModel, TransformerModel, RNNModel
from pytorch_lightning.callbacks import EarlyStopping
from torchmetrics import MeanAbsolutePercentageError

data_sets = ['vanderpol', 'fitzhugh_nagumo']
models = ['RNN', 'NBEATS', 'Transformer']
#models = ['RNN',]
results = {}

for data_name in data_sets:
    df = pd.read_csv(f'data/{data_name}_data.csv', header=None)
    #df = pd.read_csv('fitzhugh_nagumo_data.csv', header=None)
    df.columns = ['x', 'y']
    df['time'] = np.arange(0, len(df))
    # make zero mean and unit variance
    #df['x'] = (df['x'] - df['x'].mean()) / df['x'].std()
    #df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()
    series1 = TimeSeries.from_dataframe(df, 'time', ['x', 'y'])
    series_train, series_test = series1.split_after(5/6)
    #series_train, series_val = series_train.split_after(0.66)

    #DatasetLoaderCSV('vanderpol_data.csv').load()
    results[data_name] = {}
    for model_name in models:
        
        torch_metrics = MeanAbsolutePercentageError()
        my_stopper = EarlyStopping(
            monitor="val_loss",  # "val_loss",
            patience=10,
            min_delta=0.01,
            mode='min',
        )
        pl_trainer_kwargs = {"callbacks": [my_stopper]} | device_kwargs
        if model_name == 'RNN':
            model = RNNModel(input_chunk_length=1, output_chunk_length=1,
                             n_rnn_layers=1, hidden_dim=15, model='LSTM', 
                             random_state=42, n_epochs=200, 
                             pl_trainer_kwargs=pl_trainer_kwargs)
        elif model_name == 'NBEATS':
            model = NBEATSModel(input_chunk_length=1, output_chunk_length=1, random_state=42, 
                                pl_trainer_kwargs=pl_trainer_kwargs, n_epochs=200)
        elif model_name == 'Transformer':
            model = TransformerModel(input_chunk_length=1, output_chunk_length=1, random_state=42, 
                                     pl_trainer_kwargs=pl_trainer_kwargs,n_epochs=200)
        model.fit(series_train, verbose=True, val_series=series_test)
        num_points = 20
        x_range = np.linspace(-3,3,num_points)
        y_range = np.linspace(-3,3,num_points)
        mesh = np.meshgrid(x_range, y_range)
        mesh = np.stack(mesh, axis=-1)

        uv = np.zeros((num_points, num_points, 2))
        n_pred = 100
        traj = np.zeros((num_points, num_points, n_pred+1, 2))
        for i in range(num_points):
            for j in range(num_points):
                t0 = pd.DataFrame({'x': np.array([mesh[i,j,0],0]), 'y': np.array([mesh[i,j,1],0]), 'time': np.array([0,1])})
                t0_ts = TimeSeries.from_dataframe(t0, 'time', ['x', 'y'],).drop_after(1)
                #uv[i,j] = model.predict(series=t0_ts, n=1).data_array()[0,:,0] - mesh[i,j]
                pred = model.predict(series=t0_ts, n=n_pred, verbose=False, num_samples=1).data_array()[:,:,0]
                pred = np.concatenate([np.array([mesh[i,j]]), pred], axis=0)
                traj[i,j] = pred
        results[data_name][model_name] = traj
import pickle

with open('data/fig7_vector_field_darts.pkl', 'wb') as f:
    pickle.dump(results, f)
