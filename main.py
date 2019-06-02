"""
    date:   June 2019
    author: Olivier Bronchain
    contact: olivier.bronchain@uclouvain.be
    Affiliation: UCLouvain Crypto Group
"""

from Leakage_generation import *
from MI_computation import *
from Models import *

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    ###############
    print("Setting leakage parameters\n")
    #################################
    ##### THE PARAMETERS CAN BE CHANGED HERE
    D = 1
    Nk = 16
    var = 2
    serial = True
    sbox = np.random.permutation(Nk).astype(np.uint8)
    ################################

    Nd = min(1,D * serial)
    oracle = LeakageOracle(var=var,Nk=Nk,sbox=sbox,D=D,Nd=Nd)

    bin_length = 256
    Nfold = 10
    n = int(1E5/Nfold)

    ePIs = np.zeros(Nfold-1)
    gPIs = np.zeros(Nfold-1)
    eHIs = np.zeros(Nfold)

    ################
    print("\nscaling the scope ...")
    oracle.to_string()
    plaintext = np.random.randint(0,Nk,int(1E6),dtype=np.uint8)
    l = oracle.get_leakage(plaintext)
    digit = Digitizer(bin_length,np.min(l),np.max(l))

    ################
    if Nd == 1:
        print("\nComputing analytical expression")
        MI = MI_1D_integral(oracle.get_GM())

    ################
    print("\nGenerating leakage")
    hists = [Hist(Nk,Nd) for _ in range(Nfold)]
    gts = [GT(Nk,Nd) for _ in range(Nfold)]

    for i in range(Nfold):
        plaintext = np.random.randint(0,Nk,n,dtype=np.uint8)
        l,x = oracle.get_leakage(plaintext,key=0,get_intermediates=True)
        l = digit.digitize(l)
        hists[i].fit(l,x)
        gts[i].fit(l,x)

    print("\nComputing PI with %d folds"%(Nfold))
    for i in tqdm(range(Nfold),desc="fold"):
        Hist_test = hists[i]
        Hist_train  = Hist(Nk,Nd)
        gt = GT(Nk,Nd=Nd)
        for j in range(Nfold-1):
            Hist_train.merge(hists[(i+1+j)%Nfold])
            gt.merge(gts[(i+1+j)%Nfold])

            ePIs[j] += MI_sampling(Hist_test,Hist_train.to_pdfs(trick=True))
            gPIs[j] += MI_sampling(Hist_test,gt.to_GM())

    ePIs /= (Nfold)
    gPIs /= (Nfold)

    print("\nComputing HI")
    Hist_train  = Hist(Nk,Nd)
    for i in tqdm(range(Nfold),desc="HI"):
        Hist_train.merge(hists[(i)])

        eHIs[i] += MI_sampling(Hist_train,Hist_train.to_pdfs(trick=False))


    plt.figure()
    plt.loglog(np.linspace(n,n*(Nfold),len(eHIs)),eHIs,label="eHI")
    plt.loglog(np.linspace(n,n*(Nfold-1),len(ePIs)),ePIs,label="ePI")
    plt.loglog(np.linspace(n,n*(Nfold-1),len(gPIs)),gPIs,label="gPI")

    if Nd == 1:
        plt.axhline(MI,color="r",ls="--",label="MI")

    plt.grid(True,which="both")
    plt.ylabel("Information")
    plt.xlabel("# of traces to build the model")
    plt.legend()
    plt.show()
