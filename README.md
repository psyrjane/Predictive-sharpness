# Predictive-sharpness
This repository contains the code for implementing the sharpness measure from the paper "A Measure of Predictive Sharpness for Probabilistic Models". It also contains the code for reproducing the figures, tables, simulations, and experiments in the paper.

For implementation, see the folder "Measure". For reproducibility material, see the folder "Reproducibility".

Article: https://doi.org/10.48550/arXiv.2509.03309

Guide:

sharpness_multi.py - Contains the main sharpness function S(f) for continuous multidimensional pdfs, with midpoint discretization, domain preparation, and sharpness visualizations. Used for most applications.

sharpness_core.py - Contains the discrete sharpness function S(P), S_rel, domain transformations, and a basic version of the continuous sharpness function S(f).

sharpness_monte_carlo.py - Experimental function for evaluating sharpness for high-dimensional pdfs by Monte Carlo sampling.

sharpness_raw.py - Example function of calculating sharpness from raw data points by histogram evaluation.

rearranged_formulas.py - Implements the rearranged analyses for the mass-length plot with visualization.

c_plot.py - Implements the equivalent analyses from rearranged_formulas.py on concentration plots with visualization.

c_plot_multiple.py - Implements sharpness score contributions by area for multiple pdfs.

minmax_finder - Finds the min and max sharpness and entropy while the other measure is fixed. For discrete cases, also provides the minimizing or maximizing distribution.
