# PNLSS Code

This repo provides the necessary code for reproducing the results of the paper...


## Setup

GPU is required to run the codes. The following docker image is built for CUDA12. For setting up the environment we assume that you have installed Docker on your machine. First clone the repository and build the image:

```bash
git clone https://github.com/christiando/pnlss.git
cd pnlss

sudo usermod -aG docker $USER
newgrp docker
docker build -t pnlss .

git submodule update --init --recursive
```
This might take a couple of minutes. If not already installed,  install _NVIDIA Container Toolkit_ by following [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and run in the following.
```bash
sudo systemctl restart docker
```
Once this is done you can start a jupyter lab environment.

```bash
docker run --gpus all -p 8888:8888 -v $(pwd):/app/work pnlss
```

Click on the link, that is prompted in your terminal, and you can execute the notebooks that come with this repository.

## Running the codes for the figures

__Figure 2__: Run `Fig2_PNLSS-VanderPol.ipynb`
__Figure 3__: Run `Fig3_PNLSS-Lorentz_computation-time.ipynb`
__Figure 4a__: Run `Fig4a_models_Dysts-Darts.ipynb`
__Figure 4b__: Run `Fig4b_models_correlations.ipynb`
__Figure 5__: Run `Fig5_violin_plot.ipynb`
__Figure 6__: Run `Fig6_combine_plots.ipynb`
__Figure 7__: Run xxx

## Running the codes for generating figure data

The repo comes already with the data for the figures. However, we also provide the code to reproduce the data. Be aware, that this can be time intensive.

__Figure 3__: 

First run 

```bash
docker run --gpus all  -v "$(pwd)":/app pnlss python notebooks/Fig3_computation_time_nlss.py
docker run --gpus all  -v "$(pwd)":/app pnlss python notebooks/Fig3_computation_time_rbf.py
```

Then, comment out the following lines in `Fig3_compuration_time_nlss.py`

```python
# fix condition
CONV_CRIT = -1e-10 #1e-4
MAX_ITER = 50
f = open('data/lorentz_computation_time_nlss_fix.dat','wb')
```
Remove the comments at the following lines
```python
# cri condtion
# CONV_CRIT = 1e-5 #1e-4
# MAX_ITER = 100
#f = open('data/lorentz_computation_time_nlss_cri.dat','wb')
```

__Figure 4b, 5, and 6__: These figures use the fitted models and their prediction data. To re-run the benchmarking of the PNL-SS method only, run at pnlss/ (Each file takes ~3 days):

```bash
docker run --gpus all  -v "$(pwd)":/app pnlss python repos/dysts/benchmarks/compute_benchmarks_noise_fine_high.py
docker run --gpus all  -v "$(pwd)":/app pnlss python repos/dysts/benchmarks/compute_benchmarks_noise_fine_low.py
```

To compute the benchmarking of the PNL-SS and all the 13 methods in Darts, comment out the following lines in the codes and run the codes (This will take ~ 1 week):

```python
if model_name in ['LSS_Takens','NLSS_Takens']:
	print(f"{model_name} exists, but forced to re-fit")
else:
	print(equation_name + " " + model_name, flush=True)
	continue
```

The above codes use the optimized hyperparameters. To do the hyperparameter searchers of the PNL-SS and all the 13 methods in Darts, run (Each file takes ~ 3 weeks):

```bash
docker run --gpus all  -v "$(pwd)":/app pnlss python repos/dysts/benchmarks/find_hyperparameters_high.py
docker run --gpus all  -v "$(pwd)":/app pnlss python repos/dysts/benchmarks/find_hyperparameters_low.py
```