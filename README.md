# PAF - Probabilistic Analysis of Floating-point 
PAF is a prototype tool for probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with respect to standards worst-case error analysis is the user has to provide the 
probability distributions together with the input variables (thus the probabilistic in PAF), 
and a confidence interval of interest. PAF computes the roundoff error for the arithmetic expression 
conditioned on the output range landing in the given confidence interval.

# Requirements
PAF has been extensively tested on Ubuntu 16.04 running Python 3.5 (also with Python 3.7).
The global optimizer Gelpia seems to work properly only on Ubuntu 16 
(maybe also with Ubuntu 18 at your own risk), thus we inherit the same limitations in PAF.

#### Install the requirements
The script ```install``` can install all the requirements automatically.

Clone this repository. From the home directory of PAF digit:
```./install```

This is going to install:
* Python 3.7 (only if you do not have it)
* All the Python3.7 modules required in PAF (only if you do not have them already installed)
* [Gelpia](https://github.com/soarlab/gelpia/) the global optimizer
* [Z3](https://github.com/Z3Prover/z3) and [dReal](https://github.com/dreal/dreal4) (only in case they are not globally available in your OS)

# Input Language
TODO

# How to run PAF
TODO

# To CAV Artifact Evaluation Reviewers
#### Reproduce the results of Table 1
From the home directory of PAF, please run
``` ./CAV_Table_1 ```

**Note:** due to the simplicity of these benchmarks no extraordinary hardware is requested. This command can be executed on a average machine (e.g. laptop).

The results of the analysis are dumped in the folder results.
#### Reproduce the results of Table 2
From the home directory of PAF, please run
``` ./CAV_Table_2 ```

**Note:** we suggest you run this command on a machine with *at least* 32-cores, to have reasonable execution times. The execution times reported in our CAV submission where measured on a machine with 64-cores.
