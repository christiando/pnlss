from jax import config, jit
config.update("jax_enable_x64", True)
import jax
try:
    jax.devices()
except:
    config.update('jax_platforms','cpu')
    print('gpu is not recognized')
from jax import numpy as jnp

from timeseries_models.state_space_model import StateSpaceModel
from timeseries_models import state_model, observation_model
import pandas as pd
from jax import numpy as jnp


class CustomObservationModel(observation_model.LinearObservationModel):
    
    def update_hyperparameters(self, X: jnp.ndarray, smooth_dict: dict, **kwargs):
        """Update hyperparameters.

        :param smoothing_density: The smoothing density over the latent space.
        :type smoothing_density: pdf.GaussianPDF
        :param X: Observations. Dimensions should be [T, Dx]
        :type X: jnp.ndarray
        :raises NotImplementedError: Must be implemented.
        """
        #self.C = jit(self._update_C)(X, smooth_dict)
        self.Qx = jit(self._update_Qx)(X, smooth_dict)
        self.Lx = self.mat_to_cholvec(self.Qx)
        #self.d = jit(self._update_d)(X, smooth_dict)
        #self.C, self.d, self.Qx = C, d, Qx
        self.update_observation_density()

data_sets = ['vanderpol', 'fitzhugh_nagumo']
models = ['PNL-SS', 'RBF-SS']
results = {}

for data_name in data_sets:
    df = pd.read_csv(f'data/{data_name}_data.csv', header=None)
    df.columns = ['x', 'y']
    df['time'] = jnp.arange(0, len(df))
    # make zero mean and unit variance
    #df['x'] = (df['x'] - df['x'].mean()) / df['x'].std()
    #df['y'] = (df['y'] - df['y'].mean()) / df['y'].std()
    x = jnp.array(df[['x', 'y']].to_numpy())
    x_train, x_test = x[:len(x) * 5 // 6], x[len(x)* 5 // 6:]
    Dz = 2
    Dx = 2
    Dk = 5
    # Define the state model
    results[data_name] = {}
    for model_name in models:
        if model_name == 'Linear':
            sm = state_model.LinearStateModel(Dz)
        if model_name == 'PNL-SS':
            sm = state_model.LSEMStateModel(Dz, Dk)
        if model_name == 'RBF-SS':
            sm = state_model.LRBFMStateModel(Dz, Dk)
        om = CustomObservationModel(Dx, Dz)
        ssm = StateSpaceModel(state_model=sm, observation_model=om)
        ssm.fit(x_train, conv_crit=1e-4, max_iter=200)
        ssm.save(f'{data_name}_{model_name}', path='../data/')
        num_points = 20
        x_range = jnp.linspace(-3, 3, num_points)
        y_range = jnp.linspace(-3, 3, num_points)

        mesh = jnp.meshgrid(x_range, y_range)
        mesh = jnp.stack(mesh, axis=-1)

        Sigma0 = 1e-4 * jnp.eye(Dx)[None,None]
        n_pred = 100
        x_dummy = jnp.empty([1, n_pred, 2])
        traj = jnp.zeros((num_points, num_points, n_pred+1, 2))
        for i in range(num_points):
            for j in range(num_points):
                mu0 = mesh[i,j][None,None]
                prediction = ssm.predict(x_dummy, mu0=mu0, Sigma0=Sigma0)
                traj = traj.at[i,j,1:].set(prediction['x']['mu'])
                traj = traj.at[i,j,:1].set(mu0[0])
        results[data_name][model_name] = traj

import pickle

with open('data/fig7_vector_field_pnlss.pkl', 'wb') as f:
    pickle.dump(results, f)
