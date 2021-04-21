# PAF - Probabilistic Analysis of Floating-point 
PAF is a prototype tool for probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with respect to standards worst-case error analysis is the user has to provide the 
probability distributions together with the input variables (thus the probabilistic in PAF), 
and a confidence interval of interest. PAF computes the roundoff error for the arithmetic expression 
conditioned on the output range landing in the given confidence interval.

# Requirements
PAF has been extensively tested on Ubuntu 16.04 running Python 3.5. We tested PAF also with Python 3.7.
The global optimizer Gelpia seems to work properly only on Ubuntu 16 
(maybe also with Ubuntu 18 at your own risk), thus we inherit the same limitations in PAF.
PAF is entirely written in Python, thus you need Python 3.5 (or Python3.7). You also need git.

#### Install the requirements
From the home directory of PAF digit:
```./install```

This is going to install:
* Python 3.7 (only if you do not have it)
* All the Python packages required in PAF
* [Gelpia](https://github.com/soarlab/gelpia/) the global optimizer
* [Z3](https://github.com/Z3Prover/z3) and [dReal](https://github.com/dreal/dreal4) (only in case they are not available globally in your OS)

# Input Language
TODO

# To CAV Artifact Evaluation Reviewers
TODO
