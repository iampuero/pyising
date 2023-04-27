# pyising

**Ising Model Inference on Python with MCMC in C**

This project is a reimplementation of the Matlab Ising model inference with Markov Chain Monte Carlo (MCMC) using C, adapting the functionalities of the Matlab code to allow the Ising model inference in Python

The original matlab implementation can be found in [this repository](https://github.com/UFerrari/Ising_Inference) from the paper [Learning Maximal Entropy Models from finite size datasets: a fast Data-Driven algorithm allows to sample from the posterior distribution](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.94.023301), by U. Ferrari, Phys. Rev. E, 94, 2, 2018.

## Ising Model

The Ising Model is a statistical model used in physics and other fields to describe systems with binary variables. Inference involves estimating the probability distribution of the binary variables.

The probability distribution of the Ising Model can be expressed as:

$$
P(\mathbf{x}) = \frac{1}{Z} \exp(\sum_{i} h_i x_i + \sum_{i<j} J_{i,j} x_i x_j)
$$

where $\mathbf{x}$ is a vector of binary variables, $h_i$ is the external field acting on variable $i$, $J_{i,j}$ is the interaction strength between variables $i$ and $j$, and $Z$ is the partition function.

This implementation uses the Metropolis-Hastings algorithm to perform inference in the Ising Model and estimate the parameters $h$ and $J$ from binary data. 

The algorithm is implemented in C for efficiency and can be called from Python using the provided wrapper functions.

## Usage

Compile the `src/c/CMC_IsingModel.c` file.
```
gcc -shared -o MCMC_IsingModel.so -fPIC MCMC_IsingModel.c
```
