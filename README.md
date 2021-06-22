# Kernel-CLustering-with-Sigmoid-based-Regularization

## Introduction

This page contains source code for Kernel Clustering with Sigmoid-based Regularization (KCSR) for sequence segmentation. 
All the functions have been written and documented in Matlab format. Not that this implementation has no embedded C or C++ files (.mex). 
Therefore, it requires no any futher installation or compilation. However, this convenience is achieved at an expense of a slight increase in practically running time.

## Instructions

The package contains two folders and eight demo files:

   * ./data: This folder contains four datasets, inlcuding synthetic data, Weizmann data, Google spoken digits data and ordered MNIST data.
   * ./mfcc: This folder contains source code for computing del-frequency cepstral coefficents (MFCCs) from audio signals.
   * ./demoSyn_FB.m: demo of KCSR on segmentation of synthetic sequence.
   * ./demoSyn_SGA.m: demo of SKCSR on segmentation of synthetic sequence.
   * ./demoWei_FB.m:  demo of KCSR on segmentation of human action videos taken from Weizmann dataset.
   * ./demoWei_SGA.m: demo of SKCSR on segmentation of human action videos taken from Weizmann dataset.
   * ./demoGoo_FB.m:  demo of KCSR on segmentation of Google spoken digits audio.
   * ./demoGoo_SGA.m: demo of SKCSR on segmentation of Google spoken digits audio.
   * ./demoMni_SGA.m: demo of SKCSR on segmentation of ordered MNIST digists sequence.
   * ./demoMul_SGA.m: demo of MKCSR on segmentation of action video sequences of three subjects in Weizmann dataset.
