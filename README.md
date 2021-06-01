# PAF: Probabilistic Analysis of Floating-Point Computations

PAF is a prototype tool for the probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with standards worst-case error analysis is that the user has to provide 
a probability distribution for every input variable, 
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
PAF has been extensively tested only on Ubuntu 16.04 running Python 3.7 (tested also with Python 3.5).

#### Setup
The script ```install``` attempts to install all the requirements automatically.

After you clone this repository, from the home directory of PAF digit:

```./install```

The script is going to install:
* Python 3.7 (only if you do not have it). It does not overwrite your default ```python3```. In case you already have ```python3.7``` installed in your OS, the script attempts to install the required modules. You **must** have a working ```python3.7 -m pip install``` at user level (no sudo). 
* All the Python3.7 modules required in PAF (only if you do not have them already installed)
* [Gelpia](https://github.com/soarlab/gelpia/) the global optimizer
* [Z3](https://github.com/Z3Prover/z3) and [dReal](https://github.com/dreal/dreal4) (only in case they are not globally available in your OS)

# <a name="run"></a> How to run PAF
From the home directory of PAF please run

``` python3.7 src/main.py -h ``` 

This command is going to show you how to properly run PAF from the command line.

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

Please note you **must** enclose any binary operation into parenthesis (e.g. (X+Y)). 
This is because we want to be sure the user knows the order of the operations in the expression, since floating-point arithmetic does not respect the rules of real arithmetic (e.g. (X+Y)+Z != X+(Y+Z) in floating-point).

# <a name="output"></a> Output from PAF
PAF dumps the output of the analysis in the folder ```results``` in the home directory.
There is one folder for each benchmark, and the name of the folder reflects the name of the benchmark. PAF overwrites the folder in case it already exists.

Each benchmark folder contains:

* Our sound probabilistic analysis (files with 'CDF' in their name). In particular, there is a log file named 'CDF_summary' reporting the exe-time of the analysis, our sounds probabilistic range analysis, and the probabilistic error bound at the bottom of the file.
* Our unsound Monte Carlo implementation (files with 'golden' in their name). There is one file for the range and one for the error analyses.
* Two pictures comparing PAF against Monte Carlo.

# <a name="cav"></a> To CAV Artifact Evaluation Reviewers

#### Reproduce the results from the (future camera-ready) Motivation section
**Note**: at the moment our accepted paper does not include a motivation section. The camera-ready submission is going to include one, with this exact experiment.
From the home directory of PAF, please run

``` ./CAV_Motivation ```

**Note:** due to the simplicity of these benchmarks no extraordinary hardware is required. This experiment can be run on a average machine (e.g. laptop) with a reasonable execution time (≈ 3hours). The results of the analysis are dumped in the folder results.

#### Reproduce the results from the (existing) Experimental Evaluation section
From the home directory of PAF, please run

``` ./CAV_Experimental_Full ```

**Note:** we suggest you run this script on a machine with *at least* 32-cores to have reasonable execution times. The execution times reported in our CAV submission where measured on a machine with 64-cores. The expected exexution time on a 64-cores machine is about 1 week (84 benchmarks total).

#### Reproduce the results from the (existing) Experimental Evaluation section (Light Version)
From the home directory of PAF, please run

``` ./CAV_Experimental_Light ```

**Note:** this script runs only a subset of the experiments from our Experimental Evaluation section. This 'Light Version' can be run on a average machine (e.g. laptop) similarly to the previous script Motivation.

#### Run PAF on a single benchmark

From the home directory of PAF, please run

```python3.7 src/main.py  -m <mantissa> -e <exp> -d <discr_size> -tgc <timeout_cnstrs> -z -prob <confidence> <path>```

where:

*m* is the mantissa format in bits (default: 53);

*e* is the exponent format in bits (default: 11);

*d* is the size of the ds structure (default: 50);

*tgc* is the timeout in seconds for the global optimizer with constraints (default: 180);

*z* when provided means use exclusively z3 SMT solver (default: False);

*prob* is the confidence interval of interest (default: 1);

*\<path\>* is the path to the benchmark of interest (positional argument). For example, *\<path\>* can be *benchmarks/benchmarks_gaussian/Filters1_gaussian.txt*.

Please run ``` python3.7 src/main.py -h ``` for the most updated command line options.

#### Run PAF on a set of benchmarks
In case you want to run PAF on a set of benchmarks, the command line is very similar to the one for a single input program.
You just need to give the path to the folder where the input programs are, and PAF is going to process them one by one.

For example, the following command line

```python3.7 src/main.py ./benchmarks/benchmarks_gaussian/```

is going to run PAF (with default parameters) on each input program in the folder *benchmarks/benchmarks_gaussian/*.

Please run ``` python3.7 src/main.py -h ``` for the most updated command line options.

# <a name="ack"></a> Acknowledgements
We thank Ian Briggs and Mark Baranowski for their generous and prompt support with Gelpia.
We also thank Alexey Solovyev for his detailed feedback and suggestions for improvements.

PAF is supported in part by the National Science Foundation awards CCF 1552975, 1704715,  the Engineering and Physical Sciences Research Council (EP/P010040/1), and the Leverhulme Project Grant ''Verification of Machine Learning Algorithms''.
