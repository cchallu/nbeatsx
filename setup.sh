conda create --name nbeatsx python=3.7.2
source ~/anaconda3/etc/profile.d/conda.sh
#source ~/miniconda/etc/profile.d/conda.sh
conda activate nbeatsx

# basic
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0

# pytorch
#conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
#conda install pytorch==1.7.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision -c pytorch

conda install -c conda-forge jupyterlab

conda install -c conda-forge fire
pip install patool
conda install -c conda-forge tqdm
conda install -c powerai gin-config
conda install -c fastai fastcore

ipython kernel install --user --name=nbeatsx
conda install -c anaconda pylint
conda install -c anaconda pyyaml
conda install -c anaconda xlrd

conda install -c conda-forge hyperopt

# M5
conda install -c anaconda ipywidgets
conda deactivate