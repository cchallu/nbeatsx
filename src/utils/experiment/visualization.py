import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

import seaborn as sns

# Plot
def plot_prediction(y, y_hat,ax, title):
    n_y = len(y)
    n_yhat = len(y_hat)
    ds_y = np.array(range(n_y))
    ds_yhat = np.array(range(n_y-n_yhat, n_y))

    ax.plot(ds_y, y, label = 'y')
    ax.plot(ds_yhat, y_hat, label='y_hat')
    ax.set_title(title)
    
def plot_grid(x,n_row,n_col, titles=None, title_plot='plot_grid', dir='./'):
    n_graph = len(x)
    fig, axs = plt.subplots(n_row, n_col, figsize=(5*n_col, 3*n_row))
    plt.xticks(rotation=45)
    
    for i in range(n_graph):
        row = int(np.floor(i/n_col))
        col = i % n_col
        if titles is not None:
          title = titles[i]
        else:
          title = i
        plot_prediction(y=x[i][0],y_hat=x[i][1], ax=axs[row, col], title=title)
    fig_name = dir+str(title_plot)+'.png'
    plt.savefig(fig_name)
    #plt.show()