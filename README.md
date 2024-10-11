# PNLSS Code

This repo provides the necessary code for reproducing the results of the paper _A projected nonlinear state-space model for forecasting time series signals_ ([arxiv](https://arxiv.org/pdf/2311.13247)). If you are interested in only running the PNLSS model, please refer to this [repository](https://github.com/christiando/timeseries_models/blob/main/notebooks/tutorial.ipynb).


## Setup

GPU is required to run the codes. The following docker image is built for CUDA12. For setting up the environment we assume that you have installed Docker on your machine. First clone the repository and build the image:

```bash
git clone https://github.com/christiando/pnlss.git
cd pnlss
git submodule update --init --recursive

docker build -t pnlss .
```
This might take a couple of minutes. Alternatively you can also use/pull the built image from dockerhub [`chdonner/pnlss`](https://hub.docker.com/repository/docker/chdonner/pnlss/general). If not already installed,  install _NVIDIA Container Toolkit_ by following [instructions](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and run in the following.
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

__Figure 7__: Run `Fig7_vectorfield.ipynb`

## Running the codes for generating figure data

The repo comes already with the data for the figures. However, we also provide the code to reproduce the data. Be aware, that this can be time intensive. First start the docker image from within the `pnlss` folder:

```bash
docker run --gpus all -it -v "$(pwd):/app" pnlss /bin/bash
```

__Figure 3__: 

Run the following 

```bash
python notebooks/Fig3_computation_time_nlss.py
python notebooks/Fig3_computation_time_rbf.py
python notebooks/Fig3_computation_time_nlss.py --fixed
python notebooks/Fig3_computation_time_rbf.py --fixed
```

__Figure 4b, 5, and 6__: These figures use the fitted models and their prediction data. To re-run the benchmarking of the PNL-SS method only, run at pnlss/ (Each file takes ~3 days):

```bash
python repos/dysts/benchmarks/compute_benchmarks_noise_fine_high.py --pnlss_only
python repos/dysts/benchmarks/compute_benchmarks_noise_fine_low.py --pnlss_only
```

To compute the benchmarking of the PNL-SS and all the 13 methods in Darts, comment out the following lines in the codes and run the codes (This will take ~ 1 week):

```bash
python repos/dysts/benchmarks/compute_benchmarks_noise_fine_high.py
python repos/dysts/benchmarks/compute_benchmarks_noise_fine_low.py
```

The above codes use the optimized hyperparameters. To do the hyperparameter searchers of the PNL-SS and all the 13 methods in Darts, run (Each file takes ~ 3 weeks):

```bash
 repos/dysts/benchmarks/find_hyperparameters_high.py
python repos/dysts/benchmarks/find_hyperparameters_low.py
```

__Figure 7__:

Run the first file to compute the darts models for figure 7, and the second for the pnlss models.

```bash
python notebooks/Fig7_run_darts_models.py
python notebooks/Fig7_run_pnlss.py
```
