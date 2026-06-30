# Simple-Model-of-Neuron-Activation
A simulation-based inference project using BayesFlow to recover membrane time constant and resistance from Leaky Integrate‑and‑Fire neuron simulations. Includes data generation, neural posterior estimation, and diagnostics for model calibration and recovery.
Simple Model of Neuron Activation — Simulation-Based Inference
This project explores whether amortized Bayesian inference can recover key biophysical parameters of a Leaky Integrate-and-Fire (LIF) neuron model from simulated voltage traces. Using BayesFlow, a neural simulation-based inference framework, we estimate two fundamental neuronal properties:

Membrane time constant (τₘ)

Membrane resistance (R)

What the project does
Simulates LIF membrane potentials under controlled step-current stimulation.

Generates training and validation datasets using forward-Euler integration.

Trains a BayesFlow model combining:

A feed‑forward summary network (MLP)

A conditional normalizing flow for posterior estimation

Evaluates inference quality using:

Simulation-Based Calibration (SBC)

Parameter recovery plots

Posterior z‑score & contraction diagnostics

Key findings
The model successfully recovers τₘ and R with strong correlations to ground truth.

Diagnostics show informative and mostly well-calibrated posteriors.

Some biases appear at extreme parameter values due to simulation characteristics.

More expressive time-series architectures (GRU/LSTM/CNN) could further improve robustness.

Why it matters
This project demonstrates how simulation-based inference can be applied to computational neuroscience, enabling data-driven parameter estimation for simplified neuron models. It highlights both the promise of amortized inference and the importance of careful simulation design and model architecture choices.
