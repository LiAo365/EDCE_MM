# CO2-Net as Baseline

## 1. Preparation for Data and Environment

Following the origin paper [CO2-Net](https://github.com/harlanhong/MM2021-CO2-Net/). All the dataset preparation steps are the same as the origin paper.

Just download the extracted features from the [CO2_README.md](./CO2_README.md).

<!-- Out local conda environment is exported as `env.yaml` in [here](./env.yaml)** -->

Attention: Before run the training scripts, please make sure the extracted features are in the correct path. Must modify the `path-dataset` in the training scripts.

Our local platform is Ubuntu 22.04 with 2 NVIDIA RTX 4070Ti GPUs. The Pytorch version is 2.2.1, and the CUDA version is 12.1, the numpy version is 1.26.3

## 2. Training on THUMOS14

### 2.1 Baseline in local platform

We prepare the training scripts for the CO2-Net baseline in the local platform in bash file `train_thumos_baseline.sh`.

```bash
bash train_thumos_baseline.sh
```

### 2.2 Baseline + Substituted Confounder Set (SCS)

We prepare the training scripts for the CO2-Net baseline with the SCS in the local platform in bash file `train_thumos_scs.sh`.

```bash
bash train_thumos_scs.sh
```

### 2.3 Baseline + Multi-level Consistency Mining (MCM)

We prepare the training scripts for the CO2-Net baseline with the MCM in the local platform in bash file `train_thumos_mcm.sh`.

```bash
bash train_thumos_mcm.sh
```

### 2.3 Baseline + SCS + MCM

We prepare the training scripts for the CO2-Net baseline with the SCS and MCM in the local platform in bash file `train_thumos.sh`.

```bash
bash train_thumos.sh
```

## 3. Training on ActivityNet1.2

We prepare the training scripts in file `train_activitynet.sh`.

Just modify the `path-dataset` in the training scripts.

To run the baseline model, just modify the `--use_causal_intervention` and `--use_consistence_loss` to `0`.

To run baseline + SCS, just modify the `--use_causal_intervention` to `1` and `--use_consistence_loss` to `0`.

To run baseline + MCM, just modify the `--use_causal_intervention` to `0` and `--use_consistence_loss` to `1`.

To run baseline + SCS + MCM, just modify the `--use_causal_intervention` and `--use_consistence_loss` to `1`.

Noting: The hyperparameter provided in the scripts are on our local platform. You may need to adjust the hyperparameter according to your platform to achieve the best performance.

