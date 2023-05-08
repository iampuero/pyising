import ctypes
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial



# Load the shared library
mcmc_ising_model_lib = ctypes.CDLL('src/c/MCMC_IsingModel.so')

# Define the Python wrapper for the MCMC_IsingModel C function
def MCMC_IsingModel(N, B, first_steps, logTd, J, latticeIn, p, latticeOut):
    mcmc_ising_model_lib.MCMC_IsingModel.argtypes = [
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
        np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')
    ]
    
    mcmc_ising_model_lib.MCMC_IsingModel(N, B, first_steps, logTd, J, latticeIn, p, latticeOut)



def dataDriven_IsingModel(N, B, Bmin, logTd, p, chi, eta, alpha, epsThreshold, nStepMore, nWork, params, verbose):
    """
    Infer the Ising couplings via the data-driven algorithm with L2 regularization.
    
    N: Number of spins
    B: number of snapshots
    Bmin: minimal number of MCMC steps
    logTd: decorrelation time for the MCMC
    p: mean of sufficient statistics
    chi: covariance of sufficient statistics
    eta: L2 regularization parameter
    alpha: initial learning rate
    epsThreshold: early stop condition
    nStepMore: number of step after convergence
    nWork: number of parallel workers
    verbose: verbosity flag
    """

    p = p.flatten()
    D = N * (N + 1) // 2
    alphaMin = 1e-10
    alphaMax = 64.0

    time = 0

    # INITIALIZATION LATTICE
    lattice = (np.random.rand(nWork, N) < np.tile(p[:N], (nWork, 1))).astype(np.int32)

    qMat = np.zeros((nWork, D))
    # C code  MCMC_IsingModel function
    for nW in range(nWork):
        #qMat[nW, :], lattice[nW, :] = 
        MCMC_IsingModel(N, int(Bmin // nWork), int(np.log2(B) - 2), logTd, params, lattice[nW, :], params, lattice[nW, :])

    q = np.mean(qMat, axis=0)

    chiEta = chi + eta

    fOfX = -(eta * params)
    fOfX = fOfX + np.random.normal(loc=np.zeros((1, D)), scale=np.sqrt(np.diag(eta / B)))

    grad = p - q + fOfX
    contraGrad = np.linalg.solve(chiEta, grad)
    epsOpMc = np.sqrt(0.5 * B * (grad @ contraGrad) / D)
    epsOpMcOld = epsOpMc

    step = 0
    Beff = max(min(B / np.min(epsOpMcOld), B), Bmin)
    if verbose:
        print('Step - learning rate - ratio of MCMC steps - eps old - eps new')
        print(step, alpha, Beff / B, np.log(epsOpMcOld), np.log(epsOpMc))

    output = [step, alpha, epsOpMc, Beff]
    while (epsOpMc > epsThreshold) and (step < 150):

        jListTry = params + alpha * contraGrad

        qMatTry = np.zeros((nWork, D))
        for nW in range(nWork):
            qMatTry[nW, :], lattice[nW, :] = MCMC_IsingModel(N, Beff // nWork, int(np.log2(B) - 4), logTd, jListTry, lattice[nW, :], jListTry, lattice[nW, :])

        qTry = np.mean(qMatTry, axis=0)
        step += 1

        fOfX = -(eta * jListTry)
        fOfX = fOfX + np.random.normal(loc=np.zeros((1, D)), scale=np.sqrt(np.diag(eta / B)))

        gradTry = p - qTry + fOfX
        contraGradTry = np.linalg.solve(chiEta, gradTry)

        epsOpMc = np.sqrt(0.5 * B * ((p - qTry + fOfX) @ np.linalg.solve(chiEta, (p - qTry + fOfX))) / D)
        dEps = epsOpMc - epsOpMcOld
        if dEps < 0:
            alpha = min(alphaMax, alpha * 1.05)
            q = qTry
            grad = gradTry
            contraGrad = contraGradTry
            epsOpMcOld = epsOpMc
            params = jListTry
        else:
            alpha = max(alpha / np.sqrt(2.0), alphaMin)

        if verbose:
            print(step, alpha, Beff / B, np.log(epsOpMcOld), np.log(epsOpMc))

        Beff = max(min(B / (epsOpMcOld), B), Bmin)

        output.append([step, alpha, epsOpMc, Beff])

    if verbose:
        print('Inference done, now thermalization')

    alpha = alpha / 1.05
    grad = p - q + fOfX

    for ss in range(nStepMore):
        params = params + alpha * np.linalg.solve(chiEta, grad)

        qMat = np.zeros((nWork, D))
        for nW in range(nWork):
            qMat[nW, :], lattice[nW, :] = MCMC_IsingModel(N, Beff // nWork, int(np.log2(B) - 3), logTd + 1, params, lattice[nW, :])

        q = np.mean(qMat, axis=0)

        fOfX = -(eta * params)
        fOfX = fOfX + np.random.normal(loc=np.zeros((1, D)), scale=np.sqrt(np.diag(eta / B)))

        grad = p - q + fOfX
        epsOpMc = np.sqrt(0.5 * B * (grad @ np.linalg.solve(chiEta, grad)) / D)

        if epsOpMc > epsThreshold:
            alpha = alpha / np.sqrt(2.0)
        else:
            alpha = alpha * 1.05

        if verbose:
            print(step, alpha, ss, np.log(epsOpMc))

    return params, output


