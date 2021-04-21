# NBEATSx: Neural basis expansion analysis with exogenous variables
We extend the NBEATS model to incorporate exogenous factors. The resulting method, called NBEATSx, improves on a well performing deep learning model, extending its capabilities by including exogenous variables and allowing it to integrate multiple sources of useful information. 
To showcase the utility of the NBEATSx model, we conduct a comprehensive study of its application to electricity price forecasting (EPF) tasks across a broad range of years and markets. 
We observe state-of-the-art performance, significantly improving the forecast accuracy by nearly 20\% over the original NBEATS model, and by up to 5\% over other well established statistical and machine learning methods specialized for these tasks. Additionally, the proposed neural network has an interpretable configuration that can structurally decompose time series, visualizing the relative impact of trend and seasonal components and revealing the modeled processes' interactions with exogenous factors.

This repository provides an implementation of the NBEATSx algorithm introduced in [https://arxiv.org/pdf/2104.05522.pdf].
<div style="text-align:center">
<img src="./results/nbeatsx.png" width="700">
</div>

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
  journal={arXiv preprint arXiv:2104.05522},
  year={2021}
}
```
