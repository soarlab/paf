# PAF - Probabilistic Analysis of Floating-point 
PAF is a prototype tool for probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with respect to standards worst-case error analysis is the user has to provide the 
probability distributions together with the input variables (thus the probabilistic in PAF), 
and a confidence interval of interest (e.g. 0.95, 0.99). PAF computes the roundoff error for the arithmetic expression 
conditioned on the output range landing in the given confidence interval.

# Table of Contents
- [Requirements](#requirements)
- [Run PAF](#run)
- [Input Language](#input)
- [Output from PAF](#output) 
- [To CAV Artifact Evaluation Reviewers](#cav)
- [Acknowledgements](#ack)

# <a name="requirements"></a> Requirements
PAF has been extensively tested on Ubuntu 16.04 running Python 3.5 (also with Python 3.7).
The global optimizer Gelpia seems to work properly only on Ubuntu 16 
(maybe also with Ubuntu 18 at your own risk), thus we inherit the same limitations in PAF.

#### Setup
The script ```install``` can install all the requirements automatically.

After you clone this repository, from the home directory of PAF digit:
```./install```

The script is going to install:
* Python 3.7 (only if you do not have it). Do not worry, it does not overwrite your default Python3.
* All the Python3.7 modules required in PAF (only if you do not have them already installed)
* [Gelpia](https://github.com/soarlab/gelpia/) the global optimizer
* [Z3](https://github.com/Z3Prover/z3) and [dReal](https://github.com/dreal/dreal4) (only in case they are not globally available in your OS)

# <a name="run"></a> How to run PAF
From the home directory of PAF please run

``` python3.7 src/main.py -h ``` 

This command is going to show you the most updated help message about how to properly run PAF.

# <a name="input"></a> Input Language
TODO

# <a name="output"></a> Output from PAF
TODO

# <a name="cav"></a> To CAV Artifact Evaluation Reviewers
#### Reproduce the results of Table 1
From the home directory of PAF, please run
``` ./CAV_Table_1 ```

**Note:** due to the simplicity of these benchmarks no extraordinary hardware is requested. This experiment can be run on a average machine (e.g. laptop) with a reasonable execution time (≈ 3hours).

The results of the analysis are dumped in the folder results.
#### Reproduce the results of Table 2
From the home directory of PAF, please run
``` ./CAV_Table_2 ```

**Note:** we suggest you run this command on a machine with *at least* 32-cores, to have reasonable execution times. The execution times reported in our CAV submission where measured on a machine with 64-cores.

# <a name="ack"></a> Acknowledgements
TODO

