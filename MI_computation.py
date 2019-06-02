"""
    date:   June 2019
    author: Olivier Bronchain
    contact: olivier.bronchain@uclouvain.be
    Affiliation: UCLouvain Crypto Group
"""

from Leakage_generation import *
from MI_computation import *
from Models import *

import numpy as np
import scipy.stats
from scipy.integrate import quad,nquad
import math
from tqdm import tqdm

def MI_1D_integral(GMs,pr=None):
	"""
		compute the information avaible within the given gaussian mixtures. The
		returned MI is computing by integration.

		GMs: is a list of Gaussian mixtures of size Nk. Each of the GM corresponds
			to a key.
		pr: is the probability of each Gaussian Mixture (uniform if not set).

	"""

	Nk = len(GMs)
	if pr is None:
		pr = np.ones(Nk)/Nk

	HY = 0
	S = 0

	for i, _ in enumerate(tqdm(GMs,desc="Sensible varialbes")):
		HY += - pr[i] *np.log2(pr[i])
		def integrant(x):
			# computing pr_l_y
			pr_l_y = np.zeros(len(GMs))

			for j,gm in enumerate(GMs):
				pr_l_y[j] = gm.pdf(x)

			if(np.sum(pr_l_y) == 0 or pr_l_y[i] == 0):
				return 0
			else:
				pr_y_l = pr[i]*pr_l_y[i]/np.sum(pr*pr_l_y)

			ret = pr_l_y[i] * np.log2(pr_y_l)


			return ret

		q = quad(integrant,-np.inf,np.inf)
		S += pr[i] * q[0]

	return HY + S

def MI_sampling(oracle,models,nt=10000,pr=None):
	""" This function estimates the information based on the true leakage
		distribution (oracle) and its estimate (models). To compute PI, these
		two has to be different while for HI, they are the same.

        oracle: - is an histogram containing the leakge samples. Is to the true leakage
                distribution
        models: - is the model of the leakage. Is a list of objects (Hist or GM)
                implementing .pdf(x) function
	"""

	Nk = len(models)
	if pr is None:
		pr = np.ones(Nk)/Nk

	HY = 0
	S = 0

	for i, model in enumerate(models):
		HY += - pr[i] *np.log2(pr[i])

		# getting the samples from the histogram
		hist = oracle._hist[i]
		uniques = np.where(hist>0)
		oracle_pr_l_k = hist[uniques]/np.sum(hist)
		samples = np.stack(uniques).T
		nt = len(samples)
		if(nt==0):
			continue


		# Computing the model pr_K_L by Bayes law
		# getting the pr_l_x
		pr_l_k = np.zeros((Nk,nt))
		for k in range(Nk):
			pr_l_k[k,:] = models[k].pdf(samples)

		# getting the pr_k_l
		pr_k_l = pr_l_k[i,:]*pr[i]/np.sum(pr*pr_l_k.T,axis=1).T

		S += pr[i] * np.sum(np.log2(pr_k_l)*oracle_pr_l_k)

	return HY + S
