# PAF - Probabilistic Analysis of Floating-point 
PAF is a prototype tool for probabilistic analysis of round-off errors in arithmetic expressions. 
The main difference with standards worst-case error analysis is the user provides to PAF the 
probability distributions associated with the input variables (thus the probabilistic in PAF), 
and a confidence interval of interest. PAF computes the roundoff error for the expression 
conditioned on the output range landing in the given confidence interval.

# Requirements
PAF has been extensively tested on Ubuntu 16.04 running Python 3.5. 
Unfortunately, the gloabal optimizer PAF uses, Gelpia, seems to work properly only on Ubuntu 16 
(maybe also with Ubuntu 18 at your own risk), thus we inherit the same limitations.

PAF is entirely written in Python, thus you need Python 3.5 (at least). You also need git.

### Python Requirements
sudo apt-get install -y python3-tk libmpc-dev python3-dev python3-pip

### Gelpia

