"""
Original code by David Hofmann.
Translated into Python from Matlab by Thuan Tran, 2017
"""
import numpy as np

def BayesFilter(emg, prior, param):
    N = len(emg)
    dt = N * param['sf'] ** -1
    dsigma = param['pbins'][1] - param['pbins'][0]
    # time evolution update
    timeevol_prior = (dt * param['alpha'] * 
                     (np.insert(prior[0:-1], 0, prior[0]) + (-2 * self.prior) + np.append(prior[1:], prior[-1]))/pow(dsigma, 2)
                     + dt * param['beta'] + (1 + (-1* dt) * param['beta']) * self.prior)

    # measurement/observation update
    if param['model'] == 'Gauss':
        likelihood = np.divide(np.exp(np.divide(-0.5 * np.sum(np.power(emg, 2)), np.power(param['pbins'], 2))), 
                               np.power(param['pbins'], N))
    elif param['model'] == 'Laplace':
        likelihood = np.divide(np.exp(np.divide(-np.sum(np.abs(emg)), param['pbins'])), np.power(param['pbins'], N))
    else:
        print('Unknown option for likelihood model')
    posterior = np.multiply(likelihood, timeevol_prior)
    posterior = np.divide(posterior, np.sum(posterior))    # normalize posterior
    
    # point estimation
    if param['pointmax'] == True:
        index = np.argmax(posterior)
        maps = param['pbins'][index]               # maps: maximum aposteriori standard deviation
    else:
        maps = sum(posterior * param['pbins'])     # calculate expectation value
    return (maps, posterior)
