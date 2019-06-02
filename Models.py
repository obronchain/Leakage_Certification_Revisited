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
import scipy
from scipy.stats import multivariate_normal
from scipy.stats import norm


class GaussianMixture:

	def __init__(self,alpha,means,variances):
		"""
			Gaussian mixture on Nd dims

			alpha: the probability of each of the modes (Nm)
			means: Nm*Nd array with each row represention the mean of a mode
			variances: Nm*Nd*Nd, each sub Nd*Nd arrays are the varaiances of a mode
		"""

		self._alpha = np.array(alpha)
		self._means = np.array(means)
		self._variances = np.array(variances)

	def sample(self,N):
		"""
			Generates N samples from the Gaussian Mixture
		"""
		def myfunc(i):
			modes = np.random.choice(np.arange(0, len(self._alpha),dtype=int),1, p=self._alpha)[0].astype(np.int)
			samples = np.random.multivariate_normal(self._means[modes],self._variances[modes])
			return samples

		vfunc = np.vectorize(myfunc,signature="()->(k)")

		return vfunc(np.arange(N))

	def pdf(self,x):
		"""
			Returns the pdf of Gaussian Mixture at x
		"""
		pdf = 0
		for i in range(len(self._alpha)):
			pdf += self._alpha[i] * scipy.stats.multivariate_normal.pdf(x,self._means[i],self._variances[i])
		return pdf


class HistDistri():
	def __init__(self,hist,trick=False):
		self._trick=trick
		self._hist = hist #(bin_length,)*Ndim array
		self._N = np.sum(hist)
	def pdf(self,X):
		indexes = np.array_split(X,np.arange(1,len(X[0,:])),axis=1)
		A = np.squeeze(self._hist[tuple(indexes)]/self._N,axis=1)
		if self._trick:
			A[np.where(A==0)] = 1/self._N
		return A

class GT:
	""" Used to compute Gaussian Templates based on
		Observations"""

	def __init__(self,Nk=256,Nd=1):

		self._Nks = np.zeros(Nk,dtype=np.float64)
		self._sums = np.zeros((Nk,Nd),dtype=np.float64)
		self._muls = np.zeros((Nk,Nd,Nd),dtype=np.float64)
		self._Nk = Nk
		self._Nd = Nd

	def fit(self,traces,keys):
		traces = traces[:,:self._Nd]
		N = np.zeros(self._Nk)
		sums = np.zeros((self._Nk,self._Nd))
		mults = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			indexes = np.where(keys==k)[0]
			self._Nks[k] += len(indexes)
			self._sums[k,:] += np.sum(traces[indexes,:],axis=0)
			self._muls[k,:] += (np.dot(traces[indexes,:].T,traces[indexes,:]))

	def merge(self,gt):
		self._sums += gt._sums
		self._muls += gt._muls
		self._Nks += gt._Nks

	def get_template(self):
		means = np.zeros((self._Nk,self._Nd))
		vars = np.zeros((self._Nk,self._Nd,self._Nd))

		for k in range(self._Nk):
			u = self._sums[k]/self._Nks[k]

			N = self._Nd
			var = (self._muls[k,:]/self._Nks[k]) - (np.tile(u,(N,1)).T*u).T

			means[k,:] = u
			vars[k,:,:] = var

		return means,vars

	def to_GM(self):
		means,vars = self.get_template()
		alphas = np.array([1])
		GM = np.array([GaussianMixture(alphas,np.array([means[k,:]]),np.array([vars[k,:]])) for k in range(self._Nk)])
		return GM

	def input_dist(self):
		return self._Nks/np.sum(self._Nks)

class Hist:
	""" Compute histograms on the fly

		Nk: number of sensible values
		Nd: number of dimensions (typically one or two)
		bin_length: the number of bins within the histogram (i.e. 256 for an 8-bits scope)
	 """

	def __init__(self,Nk=256,Nd = 1,bin_length=256):

		bins = (range(Nk+1),)
		for i in range(Nd):
			bins = bins+(range(bin_length+1),)

		self._bins = bins
		self._hist = np.zeros((Nk,)+(bin_length,)*Nd)
		self._Nk = Nk
		self._Nd = Nd
		self._bin_length = bin_length
		self._Nks = np.zeros(Nk)	#number of samples per sensible variables
		self._Nfit = 0				#total number of samples

	def merge(self,hist):
		self._hist += hist._hist
		self._Nfit += hist._Nfit
		self._Nks += hist._Nks

	def reset(self):
		""" reset all the bins of the historams """
		self._Nfit = 0
		self._hist = np.zeros((self._Nk,)+(self._bin_length,)*self._Nd)
		self._Nks = np.zeros(self._Nk)

	def fit(self,traces,keys):
		""" adds samples to the histogram estimation

			traces: NsxNd matrix representing the leakage samples
			keys: Ndx1 matrix containing the associated sensible values to traces
		"""
		traces = np.append(keys,traces,axis=1)

		#update hist
		h,_ = np.histogramdd(traces,bins=self._bins)
		self._hist += h

		#update Nks
		unique,counts = np.unique(keys,return_counts=True)
		self._Nks[unique] += counts
		self._Nfit += len(keys)

	def hist_lin(self):
		return np.reshape(self._hist,(self._Nk,self._bin_length**self._Nd))

	def to_pdfs(self,trick=False):
		return np.array([HistDistri(self._hist[k],trick)for k in range(self._Nk)])
