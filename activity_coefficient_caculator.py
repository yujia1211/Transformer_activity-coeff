import pandas as pd
from cosmo_layer import COSMO_layer
from cosmo_layer import get_sigma_profile
import numpy as np
import os
from sklearn.model_selection import train_test_split

sigma_tabulated = np.linspace(-0.025, 0.025, 51)
compound_train_0 = list(
    pd.read_csv(r'./DataPool/compounds.csv', index_col=None, header=None)[0])
v_cosmo = pd.read_csv(r'./DataPool/compounds.csv', index_col=None, header=None)

sigma_pred = pd.read_csv(r'./hspip/pred_sigma.csv', index_col=None, header=None).values
sigma_true = pd.read_csv(r'./DataPool/sigma_true.csv', index_col=None, header=None).values

v_cosmo_test = v_cosmo.iloc[:, 1].values.reshape(-1, 1)

v_cosmo_train, v_cosmo_test = train_test_split(v_cosmo_test, test_size=0.2, random_state=33)
sigma_train, sigma_test = train_test_split(sigma_true, test_size=0.2, random_state=33)

compound_list = ['WATER', 'HEXANE', 'DIMETHYL-SULFOXIDE', 'NITROMETHANE']
for i in compound_list:
    name = r'./activity_coefficient/' + i + '/'

    if not os.path.exists(name):
        os.makedirs(name, exist_ok=True)
    if i == 'HEXANE':
        i = 'N-HEXANE'
    COSMO = COSMO_layer()

    act_pred = COSMO(input_my_sigma=sigma_pred, input_vt_sigma=get_sigma_profile(i, sigma_pred.shape[0])[0],  # 8459
                     v_compound=v_cosmo_test, v_vt=get_sigma_profile(i, sigma_pred.shape[0])[1])
    act_true = COSMO(input_my_sigma=sigma_test, input_vt_sigma=get_sigma_profile(i, sigma_test.shape[0])[0],  # 8459
                     v_compound=v_cosmo_test, v_vt=get_sigma_profile(i, sigma_test.shape[0])[1])

    pd.DataFrame(act_pred.numpy()).to_csv(name + 'pred_activity.csv', header=False, index=False)
    pd.DataFrame(act_true.numpy()).to_csv(name + 'true_activity.csv', header=False, index=False)