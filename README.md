# treetrimmer
Class project for deep learning for discrete optimization

-generateVehicleData.m
Generates the data set for vehicle hybrid powertrain problem (parametric mixed-integer quadratic program (MIQP)).
The problem matrices are taken from: https://github.com/cvxgrp/miqp_admm

-main_trainCostOutput.py
Train a neural network to estimate the optimal cost for a given power profile (parameters within the MIQP)

-main_trainDiscreteOutput.py
Train a neural network to estimate the 72 binary variables (engine on/off) for a given power profile
