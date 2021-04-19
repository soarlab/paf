# PAF - Probabilistic Analysis of Floating-point 
PAF is a prototype tool for probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with respect to standards worst-case error analysis is the user has to provide the 
probability distributions together with the input variables (thus the probabilistic in PAF), 
and a confidence interval of interest. PAF computes the roundoff error for the arithmetic expression 
conditioned on the output range landing in the given confidence interval.

# Requirements
PAF has been extensively tested on Ubuntu 16.04 running Python 3.5. 
The global optimizer Gelpia seems to work properly only on Ubuntu 16 
(maybe also with Ubuntu 18 at your own risk), thus we inherit the same limitations in PAF.

PAF is entirely written in Python, thus you need Python 3.5 (at least). You also need git.

#### Python Requirements

* ```sudo apt-get install -y python3-tk libmpc-dev python3-dev python3-pip```
* ```pip3 install -r Req_py.txt```

#### Gelpia
TODO

#### SMT solvers - Z3 and dReal
Install [Z3](https://github.com/Z3Prover/z3/) and [dReal](https://github.com/dreal/dreal4).

Make sure each solver is available globally in your system.

# Input Language
TODO

# To CAV Artifact Evaluation Reviewers
TODO
