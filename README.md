# Predictive-sharpness
This repository contains the code for implementing the sharpness measure from the paper "A Measure of Predictive Sharpness for Probabilistic Models". It also contains the code for reproducing all the figures, tables, simulations, and experiments in the paper.

For implementation, see the folder "Measure". For reproducibility material, see the folder "Reproducibility".

Article: https://doi.org/10.48550/arXiv.2509.03309

Guide:

sharpness_multi.py - Contains the main sharpness function S(f) for continuous multidimensional pdfs, with midpoint discretization, domain preparation, and sharpness visualizations. Use for most applications.

sharpness_core.py - Contains the discrete sharpness function S(P), S_rel, domain transformations, and a basic version of the continuous sharpness function S(f).
