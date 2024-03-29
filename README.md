# Kernel Clustering with Sigmoid-based Regularization

## Introduction

This page contains source code for Kernel Clustering with Sigmoid-based Regularization (KCSR) for sequence segmentation. 
All the functions have been written and documented in Matlab format. Note that this implementation has no embedded C or C++ files (.mex). 
Therefore, it requires no any further installation or compilation. However, this convenience is achieved at an expense of a slight increase in practically running time.

## Instructions

The package contains two folders and eight demo files:

   - `./data:` This folder contains four datasets, inlcuding synthetic data, Weizmann data, Google spoken digits data and ordered MNIST data.
   - `./mfcc:` This folder contains source code for computing del-frequency cepstral coefficents (MFCCs) from audio signals.
   - `./demoSyn_FB.m:` demo of KCSR on segmentation of synthetic sequence.
   - `./demoSyn_SGA.m:` demo of SKCSR on segmentation of synthetic sequence.
   - `./demoWei_FB.m:`  demo of KCSR on segmentation of human action videos taken from Weizmann dataset.
   - `./demoWei_SGA.m:` demo of SKCSR on segmentation of human action videos taken from Weizmann dataset.
   - `./demoGoo_FB.m:`  demo of KCSR on segmentation of Google spoken digits audio.
   - `./demoGoo_SGA.m:` demo of SKCSR on segmentation of Google spoken digits audio.
   - `./demoMni_SGA.m:` demo of SKCSR on segmentation of ordered MNIST digists sequence.
   - `./demoMul_SGA.m:` demo of MKCSR on segmentation of action video sequences of three subjects in Weizmann dataset.

The remaining files include: `init_g.m`, `knGauss.m`, `knLin.m`, `KCSR_balanced_FB.m`, `KCSR_balanced_Multi_SGAm.m`, `KCSR_balanced_SGAm.m`, `KCSR_balanced_SGAo.m`, `sigmoid_mixture_cutoff.m`, `sigmoid_mixture.m`, `sigmoid.m`, `bestMap.m`, `hungarian.m` and `MutualInfo.m`. They are all functional files that constitute the main implementation and evaluation of KCSR, SKCSR and MKCSR.

## Notes

1. The default parameters in the demo files are adjusted on the datasets used in the paper. You may need to adjust the parameters when applying it on a new dataset.

2. We ultilized the code `bestMap.m`, `hungarian.m` and `MutualInfo.m` provided by Deng Cai (http://www.cad.zju.edu.cn/home/dengcai/Data/Clustering.html) which is publicly available. Please check the licence of it if you want to make use of this code.

## Citations

Please cite the following paper if you use the codes:

1. Doan, Tung, and Atsuhiro Takasu. "Kernel Clustering With Sigmoid Regularization for Efficient Segmentation of Sequential Data." IEEE Access 10 (2022): 62848-62862.


