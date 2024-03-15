from tensorflow.keras import backend as K
from tensorflow.keras import layers
import pandas, numpy as np
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
import os
import subfunc_1 as sub

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# np.set_printoptions(threshold=np.inf)
class COSMO_layer(layers.Layer):
    def __init__(self, T=623.15, z_coordination=10.0, q0=79.53, r0=66.69, c_hb=85580.0, R=8.3144598,
                 sigma_hb=0.0084, EPS=3.667, AEFFPRIME=7.5, EO=2.395e-4, name=None, **kwargs):
        super(COSMO_layer, self).__init__(name=name)

        self.q0 = q0  # [A^2]
        self.r0 = r0  # [A^3]
        self.z_coordination = z_coordination
        self.c_hb = c_hb  # kcal A^4 / mol/e^2
        self.R = R / 4184  # 0.001987 # but really: 8.3144598/4184
        self.sigma_hb = sigma_hb
        self.EPS = EPS  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
        self.AEFFPRIME = AEFFPRIME
        self.EO = EO
        self.T = T
        self.x = [0.235, 1 - 0.235]

        self.FPOL = (EPS - 1.0) / (EPS + 0.5)
        self.ALPHA = (0.3 * AEFFPRIME ** (1.5)) / (EO)
        self.alpha_prime = self.FPOL * self.ALPHA

        self.sigma_tabulated = np.linspace(-0.025, 0.025, 51)
        self.sigma_m = np.tile(self.sigma_tabulated, (len(self.sigma_tabulated), 1))
        self.sigma_n = np.tile(np.array(self.sigma_tabulated, ndmin=2).T, (1, len(self.sigma_tabulated)))
        self.sigma_acc = np.tril(self.sigma_n) + np.triu(self.sigma_m, 1)
        self.sigma_don = np.tril(self.sigma_m) + np.triu(self.sigma_n, 1)
        self.DELTAW = (self.alpha_prime / 2) * (self.sigma_m + self.sigma_n) ** 2 + c_hb * K.maximum(0.0,
                                                                                                     self.sigma_acc - sigma_hb) * K.minimum(
            0.0, self.sigma_don + sigma_hb)

    def get_Gamma(self, T, psigma):
        """
        Get the value of Γ (capital gamma) for the given sigma profile
        """
        psigma = tf.reshape(psigma, [-1, 1, 51])

        Gamma = K.ones_like(psigma)
        SSS = K.exp(-self.DELTAW / (self.R * T))
        AA = SSS * psigma  # constant and can be pre-calculated outside of the loop

        for i in range(50):
            Gammanew = 1 / K.sum(AA * Gamma, axis=-1)
            Gammanew = tf.reshape(Gammanew, [-1, 1, 51])
            difference = K.abs((Gamma - Gammanew) / Gamma)
            Gamma = (Gammanew + Gamma) / 2

        return Gamma

    def get_lngamma_resid(self, A_i, T, i, psigma_mix, prof, lnGamma_mix=None):
        """
        The residual contribution to ln(γ_i)
        """
        # For the mixture
        if lnGamma_mix is None:
            lnGamma_mix = K.log(self.get_Gamma(T, np.array(psigma_mix)))
        lnGamma_mix = tf.reshape(lnGamma_mix, [-1, 51])
        # For this component
        psigma = prof / tf.reshape(A_i, [-1, 1])
        lnGammai = K.log(tf.reshape(self.get_Gamma(T, psigma), [-1, 51]))
        lngammai = A_i / self.AEFFPRIME * K.sum(psigma * (lnGamma_mix - lnGammai), axis=1)
        return lngammai

    def get_lngamma_comb(self, As, T, x, i, profs, V_COSMO_A3):
        """
        The combinatorial part of ln(γ_i)
        """
        # A = np.array([profs[0][:, :, 1].sum(axis=1), profs[1][:, :, 1].sum(axis=1)])
        As = tf.convert_to_tensor(As)

        q = As / self.q0

        r = tf.convert_to_tensor([V_COSMO_A3[0] / self.r0, (tf.reshape(V_COSMO_A3[1], [-1, 1]) / self.r0)],
                                 dtype='float32')
        r = tf.reshape(r, [2, -1])

        theta_i = x[i] * q[i] / tf.tensordot(x, q, axes=1)

        phi_i = x[i] * r[i] / tf.tensordot(x, r, axes=1)
        l = self.z_coordination / 2 * (r - q) - (r - 1)
        return (K.log(phi_i / x[i]) + self.z_coordination / 2 * q[i] * K.log(theta_i / phi_i)
                + l[i] - phi_i / x[i] * tf.tensordot(x, l, axes=1))

    def get_lngamma(self, A_i, As, T, x, i, psigma_mix, profs, V_COSMO_A3, lnGamma_mix=None):
        """
        Sum of the contributions to ln(γ_i)
        """
        return (self.get_lngamma_resid(A_i, T, i, psigma_mix, profs, lnGamma_mix=lnGamma_mix)
                + self.get_lngamma_comb(As, T, x, i, profs, V_COSMO_A3))

    def call(self, input_my_sigma, v_compound, input_vt_sigma, v_vt, **kwargs):

        # input_my_sigma = self.get_my_sigma_profile(input_my_sigma)
        input_my_sigma = tf.reshape(input_my_sigma, [-1, 51])
        input_vt_sigma = tf.convert_to_tensor(input_vt_sigma, dtype='float32')
        v_compound = tf.convert_to_tensor(v_compound)
        v_vt = tf.convert_to_tensor(v_vt)
        V_COSMO_A3 = [v_compound, v_vt]
        # profs = [input_my_sigma[0], input_vt_sigma[0]]
        As = [K.sum(input_my_sigma, axis=1), K.sum(input_vt_sigma[:, :, 1], axis=1)]

        # profs, V_COSMO_A3 = zip(*input_my_sigma, input_vt_sigma)
        # psigma_mix = sum([x[0] * profs[0][:, :, 1], x[1] * profs[1][:, :, 1]]) / sum([x[0] * As[0].reshape(-1, 1), x[1] * As[1].reshape(-1, 1)])

        psigma_mix = (self.x[0] * input_my_sigma + self.x[1] * input_vt_sigma[:, :, 1]) \
                     / (self.x[0] * tf.reshape(As[0], [-1, 1]) + self.x[1] * tf.reshape(As[1], [-1, 1]))

        lnGamma_mix = tf.reshape(K.log(self.get_Gamma(self.T, psigma_mix)), [-1, 51])
        act = self.get_lngamma(As[0], As, self.T, self.x, 0, psigma_mix, input_my_sigma, V_COSMO_A3,
                               lnGamma_mix=lnGamma_mix)

        return tf.reshape(act, [-1, 1])

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'q0': self.q0,  # [A^2]
            'r0': self.r0,  # [A^3]
            'z_coordination': self.z_coordination,
            'c_hb': self.c_hb,  # kcal A^4 / mol/e^2
            'R': self.R,  # 0.001987 # but really: 8.3144598/4184
            'sigma_hb': self.sigma_hb,
            'EPS': self.EPS,  # (LIN AND SANDLER USE A CONSTANT FPOL WHICH YIELDS EPS=3.68)
            'AEFFPRIME': self.AEFFPRIME,
            'EO': self.EO,
            'T': self.T})
        return config


# DIMETHYL-SULFOXIDE
# HEXANE
# NITROMETHANE
# WATER

sigma_tabulated = np.linspace(-0.025, 0.025, 51)

def get_sigma_profile(name, rep):
    df = pandas.read_csv(
        r'./profiles/VT2005/Sigma_Profile_Database_Index_v2.txt',
        sep='\t')
    mask = df['Compound Name'] == name
    assert (sum(mask) == 1)
    index = int(df[mask]['Index No.'].iloc[0])
    V_COSMO = float(df[mask]['Vcosmo, A3'].iloc[0])
    V_COSMO = np.tile(V_COSMO, rep)
    with open(
            r'./profiles/VT2005/Sigma_Profiles_v2/VT2005-' + '{0:04d}'.format(
                    index) + '-PROF.txt') as fp:
        dd = pandas.read_csv(fp, names=['sigma [e/A^2]', 'p(sigma)*A [A^2]'], sep='\s+')
        dd['A'] = dd['p(sigma)*A [A^2]'].sum()
        dd['p(sigma)'] = dd['p(sigma)*A [A^2]'] / dd['A']
        dd = pandas.concat([dd] * rep, ignore_index=True).values.reshape(rep, 51, -1)
        return dd, V_COSMO


