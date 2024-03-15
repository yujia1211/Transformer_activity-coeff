import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split

compound_list = ['WATER', 'HEXANE', 'DIMETHYL-SULFOXIDE','NITROMETHANE']

for i in compound_list:

    name = './activity_coefficient/' + i + '/'

    pred = pd.read_csv(name + r'/pred_activity.csv', header=None,
                       index_col=None).values

    pred = np.exp(pred)

    true = pd.read_csv(name + r'/true_activity.csv', header=None,
                       index_col=None).values

    true = np.exp(true)

    max_v = max(true.max(), pred.max())
    min_v = min(pred.min(), true.min())

    x = np.linspace(min_v, max_v)

    r2_test = r2_score(true, pred)
    figure = plt.figure()
    plt.plot(x, x, color='black', zorder=0)
    plt.scatter(true, pred, c='r', marker='o', zorder=1, alpha=0.2, s=5,
                label='Test')

    plt.xlim(min_v, max_v)
    plt.ylim(min_v, max_v)
    plt.xlabel('true activity coefficient')
    plt.ylabel('model predict activity coefficient')
    plt.title(
             'Transformer SMILES R2 = %.04f' % r2_test)
    plt.legend(fontsize=8)
    plt.show()
