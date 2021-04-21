# NBEATSx: Neural basis expansion analysis with exogenous variables
We extend the NBEATS model to incorporate exogenous factors. The resulting method, called NBEATSx, improves on a well performing deep learning model, extending its capabilities by including exogenous variables and allowing it to integrate multiple sources of useful information. 
To showcase the utility of the NBEATSx model, we conduct a comprehensive study of its application to electricity price forecasting (EPF) tasks across a broad range of years and markets. 
We observe state-of-the-art performance, significantly improving the forecast accuracy by nearly 20\% over the original NBEATS model, and by up to 5\% over other well established statistical and machine learning methods specialized for these tasks. Additionally, the proposed neural network has an interpretable configuration that can structurally decompose time series, visualizing the relative impact of trend and seasonal components and revealing the modeled processes' interactions with exogenous factors.

This repository provides an implementation of the NBEATSx algorithm introduced in [https://arxiv.org/pdf/2104.05522.pdf].
<div style="text-align:center">
<img src="./results/nbeatsx.png" width="700">
</div>

## Electricity Price Forecasting Results

| Unnamed: 0   |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   INBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|-------------:|
| MAE          |  2.28 |    2.11 |     2.11 |  2.11 |   1.95 |  1.71 |        1.65 |         1.68 |
| rMAE         |  0.72 |    0.67 |     0.67 |  0.67 |   0.62 |  0.54 |        0.52 |         0.53 |
| sMAPE        |  6.51 |    6.09 |     6.06 |  6.1  |   5.62 |  4.97 |        4.83 |         4.89 |
| RMSE         |  4.08 |    3.92 |     3.98 |  3.89 |   3.6  |  3.36 |        3.27 |         3.33 |
| <span style="color: red;">text</span> | | | | | | | | |
| MAE          |  3.88 |    3.63 |     3.48 |  3.68 |   3.09 |  3.07 |        3.02 |         3.01 |
| rMAE         |  0.8  |    0.75 |     0.72 |  0.76 |   0.64 |  0.63 |        0.62 |         0.62 |
| sMAPE        | 14.66 |   14.26 |    13.56 | 14.09 |  12.54 | 12    |       11.97 |        11.91 |
| RMSE         |  6.26 |    5.87 |     5.59 |  5.94 |   5.14 |  5.2  |        5.06 |         5    |
| MAE          |  7.04 |    7.01 |     6.83 |  7.05 |   6.59 |  6.07 |        6.14 |         6.17 |
| rMAE         |  0.86 |    0.86 |     0.83 |  0.86 |   0.8  |  0.74 |        0.75 |         0.75 |
| sMAPE        | 16.29 |   15.95 |    16.03 | 16.21 |  15.95 | 14.11 |       14.68 |        14.52 |
| RMSE         | 17.25 |   16.76 |    16.99 | 17.07 |  16.29 | 15.95 |       15.46 |        15.43 |
| MAE          |  4.74 |    4.68 |     4.79 |  4.85 |   4.25 |  4.06 |        3.98 |         3.97 |
| rMAE         |  0.8  |    0.78 |     0.8  |  0.81 |   0.71 |  0.68 |        0.67 |         0.67 |
| sMAPE        | 13.49 |   13.25 |    13.62 | 13.41 |  13.25 | 11.49 |       11.07 |        11.29 |
| RMSE         | 13.68 |   11.89 |    12.09 | 13.78 |  10.76 | 11.77 |       11.61 |        11.08 |
| MAE          |  5.73 |    5.64 |     5.37 |  4.58 |   4.11 |  3.59 |        3.46 |         3.37 |
| rMAE         |  0.71 |    0.7  |     0.67 |  0.57 |   0.51 |  0.45 |        0.43 |         0.42 |
| sMAPE        | 21.22 |   21.09 |    19.71 | 18.52 |  16.98 | 14.68 |       14.78 |        14.34 |
| RMSE         |  9.39 |    9.17 |     9.03 |  7.69 |   6.99 |  6.08 |        5.84 |         5.64 |


### Run NBEATSx experiment from console
```console
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python src/hyperopt_nbeatsx.py --dataset 'NP' --space "nbeats_x" --data_augmentation 0 --random_validation 0 --n_val_weeks 52 --hyperopt_iters 1500 --experiment_id "20210129_0_0"
```

## Citation

If you use NBEATSx, please cite the following paper:

```console
@article{olivares2021neural,
  title={Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx},
  author={Olivares, Kin G and Challu, Cristian and Marcjasz, Grzegorz and Weron, Rafa{\l} and Dubrawski, Artur},
  journal = {International Journal of Forecasting, submitted},
  volume = {Working Paper version available at arXiv:2104.05522},
  year={2021}
}
```
