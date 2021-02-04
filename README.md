# NBEATSx: Exogenous Variables Generalization of the Neural basis expansion analysis
In this work we present the neural basis expansion analysis with exogenous variables (NBEATSx) that focuses on solving the univariate times series forecasting problem. The NBEATSx improves on a well performing deep learning model, extending its capabilities towards exogenous variables and allowing it to integrate different sources of useful information. To showcase the use of the NBEATSx model we use available datasets from Electricity Price Forecasting (EPF) tasks across a broad range of years and markets where exogenous variables have proven fundamental.

<div style="text-align:center">
<img src="./results/nbeatsx.png" width="700">
</div>

### Run NBEATSx experiment from console
```console
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/hyperopt_nbeatsx.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 1500 --experiment_id "20210129_0_0"
```

## REFERENCES
1. [N-BEATS: Neural basis expansion analysis for interpretable time series forecasting](https://arxiv.org/abs/1905.10437)
