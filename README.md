TDAN — Temporal Deformable Attention Network for Traffic/Time-Series Prediction

TDAN
A repository implementing a Temporal Deformable Attention Network (TDAN) for time-series forecasting and/or congestion prediction. This project contains data preprocessing, model training, evaluation, and inference code, plus utilities to visualize results and export models for deployment.

Project overview

TDAN is a temporal attention based model that learns deformable attention patterns over time — i.e., it can focus on irregularly spaced or temporally-shifted patterns for forecasting tasks. In this repository TDAN is adapted to traffic/vehicle-count/congestion forecasting problems (but the codebase is modular and can be used for generic univariate/multivariate time-series forecasting).

Use cases:

Short-term traffic/vehicle-count forecasting at junctions

Congestion level nowcasting using multi-modal inputs

Feeding predictions into adaptive traffic control or routing systems
