# NBEATSx: Neural basis expansion analysis with exogenous variables
We extend the NBEATS model to incorporate exogenous factors. The resulting method, called NBEATSx, improves on a well performing deep learning model, extending its capabilities by including exogenous variables and allowing it to integrate multiple sources of useful information. 
To showcase the utility of the NBEATSx model, we conduct a comprehensive study of its application to electricity price forecasting (EPF) tasks across a broad range of years and markets. 
We observe state-of-the-art performance, significantly improving the forecast accuracy by nearly 20\% over the original NBEATS model, and by up to 5\% over other well established statistical and machine learning methods specialized for these tasks. Additionally, the proposed neural network has an interpretable configuration that can structurally decompose time series, visualizing the relative impact of trend and seasonal components and revealing the modeled processes' interactions with exogenous factors.

This repository provides an implementation of the NBEATSx algorithm introduced in [https://arxiv.org/pdf/2104.05522.pdf].
<div style="text-align:center">
<img src="./results/nbeatsx.png" width="700">
</div>

## Electricity Price Forecasting Results

Nord Pool
| METRIC       |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   NBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|------------:|
| MAE          |  2.26 |    2.09 |     2.08 |  2.01 |   1.74 |  1.68 |        1.58 |        1.62 |
| rMAE         |  0.71 |    0.66 |     0.66 |  0.63 |   0.55 |  0.53 |        0.5  |        0.51 |
| sMAPE        |  6.47 |    6.04 |     5.96 |  5.84 |   5.01 |  4.88 |        4.63 |        4.7  |
| RMSE         |  4.08 |    3.89 |     3.94 |  3.71 |   3.36 |  3.32 |        3.16 |        3.27 |

Pennsylvania-New Jersey-Maryland
| METRIC       |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   NBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|------------:|
| MAE          |  3.83 |    3.59 |     3.49 |  3.53 |   3.01 |  2.86 |        2.91 |        2.9  |
| rMAE         |  0.79 |    0.74 |     0.72 |  0.73 |   0.62 |  0.59 |        0.6  |        0.6  |
| sMAPE        | 14.5  |   14.12 |    13.57 | 13.64 |  11.98 | 11.33 |       11.54 |       11.61 |
| RMSE         |  6.24 |    5.83 |     5.64 |  5.74 |   5.13 |  5.04 |        5.02 |        4.84 |

European Power Exchange Belgium
| METRIC       |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   NBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|------------:|
| MAE          |  7.2  |    6.96 |     6.84 |  7.19 |   6.14 |  5.87 |        5.96 |        6.11 |
| rMAE         |  0.88 |    0.85 |     0.83 |  0.88 |   0.75 |  0.72 |        0.73 |        0.75 |
| sMAPE        | 16.26 |   15.84 |    15.8  | 16.11 |  14.55 | 13.45 |       13.86 |       14.02 |
| RMSE         | 18.62 |   16.84 |    17.13 | 18.07 |  15.97 | 15.97 |       15.76 |       15.8  |

European Power Exchange France
| METRIC       |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   NBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|------------:|
| MAE          |  4.65 |    4.65 |     4.74 |  4.56 |   3.98 |  3.87 |        3.81 |        3.79 |
| rMAE         |  0.78 |    0.78 |     0.8  |  0.76 |   0.67 |  0.65 |        0.64 |        0.64 |
| sMAPE        | 13.03 |   13.22 |    13.3  | 12.7  |  11.57 | 10.81 |       10.59 |       10.69 |
| RMSE         | 13.89 |   11.83 |    12.01 | 12.94 |  10.68 | 11.87 |       11.5  |       11.25 |

European Power Exchange Germany
| METRIC       |    AR |   ESRNN |   NBEATS |   ARX |   LEAR |   DNN |   NBEATSx-G |   NBEATSx-I |
|:-------------|------:|--------:|---------:|------:|-------:|------:|------------:|------------:|
| MAE          |  5.74 |    5.6  |     5.31 |  4.36 |   3.96 |  3.41 |        3.31 |        3.29 |
| rMAE         |  0.71 |    0.7  |     0.66 |  0.54 |   0.49 |  0.42 |        0.41 |        0.41 |
| sMAPE        | 21.37 |   20.97 |    19.61 | 17.73 |  15.75 | 14.08 |       13.99 |       13.99 |
| RMSE         |  9.63 |    9.09 |     8.99 |  7.38 |   7.08 |  5.93 |        5.72 |        5.65 |


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
