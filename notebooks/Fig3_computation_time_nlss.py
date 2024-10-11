#CUDA_VISIBLE_DEVICES=0 python computation_time_nlss.py 

import numpy
import matplotlib.pyplot as plt

import sys
import os

# Calculating computatin time
import time

from jax import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp

# from gaussian_toolbox import timeseries
# from gaussian_toolbox.timeseries import state_model, observation_model, ssm
import gaussian_toolbox
from timeseries_models import state_model, observation_model, state_space_model
import argparse

from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D

from scipy.integrate import solve_ivp
from scipy.stats import zscore
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--fixed', action='store_true', help='Run fixed condition')
args = parser.parse_args()


# Settings
# Noise
numpy.random.seed(0)
noise_level = .01 #1.

#EM convergence criteria

if args.fixed:
    # fix condition
    CONV_CRIT = -1e-10 #1e-4
    MAX_ITER = 50
else:
    # cri condtion
    CONV_CRIT = 1e-5 #1e-4
    MAX_ITER = 100




REPEAT = 10
Dk_range = range(5,41,5)
computation_time = numpy.zeros((REPEAT,len(Dk_range)))
dimension = numpy.zeros((REPEAT,len(Dk_range)))
likelihood = numpy.zeros((REPEAT,len(Dk_range)))
param_list = numpy.zeros((REPEAT,len(Dk_range)))


for repeat in range(REPEAT):
    print(repeat)


    # Data generation
    rho = 28.0
    sigma = 10.0
    beta = 8.0 / 3.0

    def lorenz(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

    state0 = [1.0, 1.0, 1.0]
    t = numpy.linspace(0, 30, 300) #short trial

    states = odeint(lorenz, state0, t)

    X = states

    X = zscore(X, axis=0)
    X0 = X.copy()
    X += noise_level * numpy.random.randn(*X.shape)


    # Model fitting
    import numpy
    numpy.random.seed(0)
    import matplotlib.pyplot as plt
    import sys
    import os

    # from jax.config import config
    # config.update("jax_enable_x64", True)
    from jax import numpy as jnp

    # from gaussian_toolbox import timeseries
    # from gaussian_toolbox.timeseries import state_model, observation_model, ssm
    import gaussian_toolbox
    from timeseries_models import state_model, observation_model, state_space_model

    Dx = 3
    Dz = 3

    k = 0
    for Dk in Dk_range:
        print(Dk)
        param_list[repeat][k]=Dk
        
        # NLSS
        om = observation_model.LinearObservationModel(Dx, Dz)
        #om.pca_init(X)
        sm = state_model.LSEMStateModel(Dz, Dk)
        
        #ssm_emd = state_space_model.StateSpaceModel(jnp.array(X), observation_model=om, state_model=sm, max_iter=MAX_ITER, conv_crit=CONV_CRIT) 
        ssm_emd = state_space_model.StateSpaceModel(observation_model=om, state_model=sm)
        
        tc = time.time()
        llk_list, p0_dict, smooth_dict, two_step_smooth_dict = ssm_emd.fit(X, max_iter=MAX_ITER, conv_crit=CONV_CRIT)
        
        computation_time[repeat][k] = (time.time() - tc)
        print('lorentz in %f s' %(time.time() - tc))
        
        likelihood[repeat][k] = llk_list[-1]
        
        #state 
        A_dim = numpy.prod(ssm_emd.sm.A.shape) 
        b_dim = numpy.prod(ssm_emd.sm.b.shape)
        W_dim = numpy.prod(ssm_emd.sm.W.shape)
        
        tmp = ssm_emd.sm.Qz.shape[0]
        tmp = int(tmp + tmp*(tmp-1)/2)
        Qz_dim = tmp
        
        DIM = A_dim + b_dim + W_dim + Qz_dim
        
        # observation
        C_dim = numpy.prod(ssm_emd.om.C.shape) 
        d_dim = numpy.prod(ssm_emd.om.d.shape)
        tmp = ssm_emd.om.Qx.shape[0]
        tmp = int(tmp + tmp*(tmp-1)/2)
        Qx_dim = tmp
        
        DIM += C_dim + d_dim + Qx_dim
        dimension[repeat][k]=DIM

        k += 1


# Saving the data
if args.fixed:
    f = open('data/lorentz_computation_time_nlss_fix.dat','wb')
else:
    f = open('data/lorentz_computation_time_nlss_cri.dat','wb')
pickle.dump((param_list, computation_time, dimension, likelihood), f)
f.close
