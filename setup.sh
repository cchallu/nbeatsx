conda create --name nbeatsx_epf python=3.7.2
source ~/anaconda3/etc/profile.d/conda.sh
conda activate nbeatsx_epf

# basic
conda install -c anaconda numpy==1.16.1
conda install -c anaconda pandas==0.25.2
conda install -c conda-forge matplotlib==3.1.1
conda install -c anaconda seaborn==0.9.0
conda install -c anaconda scipy==1.5.2

# pytorch
#conda install pytorch==1.7.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
conda install pytorch torchvision -c pytorch

# other
conda install -c conda-forge jupyterlab
conda install -c conda-forge tqdm
conda install -c conda-forge hyperopt
conda install -c anaconda requests

ipython kernel install --user --name=nbeatsx_epf

#pip install patool

conda install -c anaconda ipywidgets
conda deactivate