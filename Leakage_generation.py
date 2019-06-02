"""
    date:   June 2019
    author: Olivier Bronchain
    contact: olivier.bronchain@uclouvain.be
    Affiliation: UCLouvain Crypto Group
"""

import numpy as np
from scipy.stats import multivariate_normal
from Leakage_generation import *
from MI_computation import *
from Models import *

def get_HW(tabular):
        """ return the HW of 8bits values array """
        HW_val = np.array([bin(n).count("1") for n in range(0,256)],dtype=int)
        return HW_val[tabular]

class Digitizer:
    """
        This class is used to make the leakage function discrete
    """
    def __init__(self,nbins,min,max):
        self._nbins = nbins
        self._min = min
        self._max = max

    def digitize(self,X):
        return np.digitize(np.minimum(X,self._max),np.linspace(self._min,self._max,self._nbins-1))


class LeakageOracle:
    def __init__(self,var,Nk,model=None,
            sbox=None,D=1,
            serial=False,secret_key=None,
            Nd=1,op=np.bitwise_xor):
        """
            LeakageOracle generate simulated leakage samples according to multiple parameters:

            Nd:     Number of dimension in the leakage. Default set to Nsbox x (serial ? D:1)
            Nk:     Numer of possible intermediate sensible values
            model:  Leakage model (mean) associated to a given intermediate value X.
                    If is an array of size (Nk,), same leakage model is used at every point
                    If is an array of size (Nk,Nd), leakage model varies as function of Nd.
                    Default model is HW.
            var:    Covariance matrix for the Gaussian noise in the leakage samples and is
                    so of size NdxNd. If set to a constant, the noise is assumed to be independent
                    on each dimension with variance var.
            D:      Number of shares within the implementation (boolean masking)
            sbox:   The permuation applied to intermediate values.
                    If of size (Nk,Nsbox), each sensible variables is passed into Nsbox permuations
                    If of size (Nk,), assumed to be of size (Nk,1).
                    Default Sbox is a random permuation.
            serial: Shares processed serially or not. If True, the leakage of each share is
                    contained in independent time samples. If False, the leakage of each shares
                    are summed together on a same point in time.
            op:     Operation mixing to key and the plaintext. Usually set to xor.
        """

        self._Nk = Nk

        if model is None:
            model = get_HW(np.arange(Nk))
        self._model = model

        if sbox is None:
            sbox = np.random.permutation(Nk).astype(int)

        if serial:
            self._Ndim_use = Nd*D
        else:
            self._Ndim_use = Nd

        self._Ndim = self._Ndim_use

        self._sbox = sbox
        self._D = D
        self._indep_noise =  not (type(var) == np.ndarray)
        self._serial = serial
        self._op = op

        if secret_key is None:
            secret_key = np.random.randint(0,Nk,dtype=int)

        self._secret_key = np.array(secret_key)
        self._Nd = Nd

        if type(var) != np.ndarray:
            var = np.eye(self._Ndim)*var
            indep_noise = True

        self._var = var

    def get_leakage(self,plaintext,key=None,get_intermediates=False):
        """
        args :
            plaintext   : the plaintext of the implementation. Is a (N,) array.
            key         : is the key used. If not specified, self._secret_key is used. if set
                            to "random", a random key is used for each encryption.

        return :
            leakage     : return leakage samples
            data_clear  : intermediate values corresponding the leakage samples

        """

        if np.ndim(plaintext) == 1 and np.ndim(self._sbox) == 1:
            plaintext = np.reshape(plaintext,plaintext.shape+(1,))

        if type(key) == np.ndarray and np.ndim(key) == 1 and np.ndim(self._sbox) == 1:
            key = np.reshape(key,key.shape+(1,))

        ## building the data manipulated in clear
        if key is None:
            key = self._secret_key
        elif key == "random":
            key = np.random.randint(0,self._Nk,plaintext.shape)

        data_clear = self._sbox[self._op(plaintext,key)]

        ## applying sharing
        if self._D > 1:
            shape = data_clear.shape + (self._D-1,)
            R = np.random.randint(0,self._Nk,shape).T

            R_s = np.zeros(data_clear.shape,dtype=np.int).T
            for d in range(self._D-1):
                R_s = np.bitwise_xor(R_s,R[d,:])

            R_s = R_s.T
            R = R.T
            data = np.reshape(np.bitwise_xor(R_s,data_clear),data_clear.shape+(1,))
            data = np.concatenate((data,R),axis=2)
        else:
            data = data_clear

        ## applying leakage model
        leakage_i = self._model[data]

        ## ordering the leakages
        if self._D > 1:
            if self._serial:
                shape = list(leakage_i.shape[:2])
                shape[1] *= self._D
                shape = tuple(shape)
                leakage_i = np.reshape(leakage_i,shape)
            else:
                leakage_i = np.sum(leakage_i,axis=2)

        leakage = np.zeros((len(plaintext),self._Ndim))
        leakage[:,:self._Ndim_use] = leakage_i.astype(np.float32)

        ## adding noise
        N = len(leakage)

        if self._indep_noise:
            leakage += np.random.normal(0,np.sqrt(np.diag(self._var)),size=leakage.shape)
        else:
            leakage += np.random.multivariate_normal(np.zeros(len(leakage[0,:])),cov=self._var,size=len(leakage[:,0]))

        if get_intermediates:
            return leakage,self._op(plaintext,key).reshape(len(plaintext),1)
        else:
            return leakage

    def get_GM(self):
        """
            return Gaussian mixture for all the possible plaintext
            values (key set to 0) accross all the sboxes
            This is used to compute the mutual information with integration
        """
        D = self._D
        Nk = self._Nk
        Nd = self._Nd
        Ndim = self._Ndim
        sbox = self._sbox

        GMs = np.array([None for _ in range(Nk)])
        Nmodes = Nk**(D-1)

        # getting GM for each possible intermediate values
        for k in range(Nk):

            # all possible randomness
            R = np.zeros((D,Nmodes),dtype=int)
            for i in range(D-1):
                r = np.repeat(np.arange(Nk),self._Nk**(self._D-2-i))
                R[i,:] = np.tile(np.repeat(np.arange(Nk),self._Nk**(self._D-2-i)),Nk**i)

            R_s = np.zeros(Nmodes,dtype=int) + k
            for i in range(D-1):
                R_s = np.bitwise_xor(R_s,R[i,:])

            R[-1,:] = R_s

            # applying model to shares
            leakage = self._model[R]

            # reducing dimension if parallel implem
            if not self._serial:
                leakage = np.sum(leakage,axis=0,keepdims=True)

            # get means and alpha
            means,alphas = np.unique(leakage,axis=1,return_counts=True)
            alphas = alphas/np.sum(alphas)
            variances = np.tile(self._var,(len(alphas),1,1))
            GMs[k] = GaussianMixture(alphas,means.T,variances)

        if sbox.ndim > 1:
            # computing the GM for the plaintexts
            GMs_traces = np.array([None for _ in range(Nk)])

            for p in range(Nk):
                GMs_p = GMs[sbox[p]] # GM at all the sbox outputs

                # number of modes at the ooutput of each sbox
                modes = np.zeros(len(GMs_p),dtype=int)
                for i,gms_p in enumerate(GMs_p):
                    modes[i] = len(gms_p._alpha)

                # modes in the entire traces
                Nmodes = int(np.prod(modes))
                alphas = np.ones(Nmodes)
                means = np.zeros((Nmodes,Ndim))
                variances = np.tile(self._var,(Nmodes,1,1))

                x = int(Ndim/Nd)
                for i,gms_p in enumerate(GMs_p):
                    n_repeat = int(Nmodes/np.prod(modes[:i+1]))
                    n_tile = int(Nmodes/np.prod(modes[i:]))

                    r = np.repeat(gms_p._means,n_repeat,axis=0)
                    r = np.tile(r,(n_tile,1))
                    means[:,x*i:x*(i+1)] = r

                    r = np.repeat(gms_p._alpha,n_repeat)
                    r = np.tile(r,n_tile)
                    alphas[:] *= r

                GMs_traces[p] = GaussianMixture(alphas,means,variances)
        else:
            GMs_traces = GMs[sbox[np.arange(Nk)]]

        GMs = GMs_traces

        return GMs

    def to_string(self):
        print("--Leakage oracle overview--")
        print("Number of dim in leakage traces: %d"%(self._Ndim))
        print("Number of shares: %d"%(self._D))
        print("Shares manipulated in serial %d"%(self._serial))
        print("SNR: %f"%(np.var(self._model)/self._var[0,0]))
