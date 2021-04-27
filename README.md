# PAF - Probabilistic Analysis of Floating-point arithmetic
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

This command is going to show you the most updated *help message* with the description of the input parameters to properly run PAF from the command line.

# <a name="input"></a> Input Programs

**Note**: in the folder benchmarks in the home directory, you can find many and many valid input programs you can run/modify based on your needs.

The input files for PAF are txt files with the following format:

``` Line 1 - Variables Declarations ``` 

``` Line 2 - Arithmetic Expression ```

## Variables Declarations

Variables declarations is a list of declarations, separated by a comma, where each declaration has the following format:

``` <variable-name>:<distribution>(<lower_bound>, <upper_bound>) ```

where:

``` <variable-name> ``` is the alphabetic name of the variable.

``` <distribution> ``` is the distribution of the variable. At the moment PAF supports **U** (uniform), **N** (standard-normal), **E** (exponential) and **R** (rayleigh) distributions.

``` <lower_bound> ``` and ```<upper_bound>``` are the numeric bounds where you want to truncate the support of the distribution of choise.

## Arithmetic Expression
The arithmetic expression is the function you want to analyze with PAF. Clearly, the function has to be expressed in terms of the input variables declared in the previous step. The only exception are constants (you do not need to declare constants). 

At the moment PAF supports the 4 basic arithmetic operations (+,-,\*,/).

Please note in your arithmetic expression you **must** enclose any binary operation into parenthesis (e.g. (X+Y)). 
This is because we want to be sure the user knows the order of the operations in the expression, since floating-point arithmetic does not respect real arithmetic (e.g. (X+Y)+Z != X+(Y+Z) in floating-point).

# <a name="output"></a> Output from PAF
TODO

# <a name="cav"></a> To CAV Artifact Evaluation Reviewers

#### Reproduce the results from the (future) Motivation section
**Note**: at the moment our accepted paper does not include a motivation section. The camera-ready submission is going to include one, with this exact experiment.
From the home directory of PAF, please run

``` ./CAV_Table_1 ```

**Note:** due to the simplicity of these benchmarks no extraordinary hardware is requested. This experiment can be run on a average machine (e.g. laptop) with a reasonable execution time (≈ 3hours).

The results of the analysis are dumped in the folder results.

#### Reproduce the results from the (existing) Experimental Evaluation section
From the home directory of PAF, please run

``` ./CAV_Table_2 ```

**Note:** we suggest you run this command on a machine with *at least* 32-cores to have reasonable execution times. The execution times reported in our CAV submission where measured on a machine with 64-cores.

#### Run PAF on a single benchmark
From the home directory of PAF, please run

```python3 src/main.py -m 24 -e 8 -d 50 -prob 0.99 <benchmark-path>```

where *m* is the mantissa format (in bits), *e* is the exponent format (in bits), *d* is the size of the ds structure (discretization), and *prob* is the confidence interval of interest. *<benchmark-path>* is the path to the benchmark of interest.
  
For example, <benchmark-path> can be *benchmarks/benchmarks_gaussian/Filters1_gaussian.txt*.

#### Run PAF on a set of benchmarks
In case you want to run PAF on a set of benchmarks, the command line is very similar to the one for a single input program.
You just need to give the path to the folder where the input programs are, and PAF is going to process them one by one.

For example, the following command line

```python3 src/main.py -m 24 -e 8 -d 50 -prob 0.99 ./benchmarks/benchmarks_gaussian/```

is going to run PAF on each input program in the folder *benchmarks/benchmarks_gaussian/*.

# <a name="ack"></a> Acknowledgements
TODO

