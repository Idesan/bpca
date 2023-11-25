# bpca

This repo presents a Python implementation of the **Adaptation-Classification Framework** introduced by

> Takumi Kitamoto, Tsuyoshi Idé, Yuta Tezuka, Norio Wada, Yui Shibayama, Yuya Tsurutani, Tomoko Takiguchi, Sachiko Suematsu, Kei Omata, Yoshikiyo Ono, Ryo Morimoto, Yuto Yamazaki, Jun Saito, Hironobu Sasano, Fumitoshi Satoh, and Tetsuo Nishikawa, Identifying Primary Aldosteronism Patients who Require Adrenal Venous Sampling: A Multi-center Study, Scientific Reports, to appear.

The primary goal of this framework is to identify primary aldosteronism patients who could benefit from specific surgical treatment.

## Problem setting 

As the name implies, the proposed framework comprises two modules:
- Data adaptation module,
- Patient classification module.

The key assumption is that it operates in a **multicenter** setting. In other words, it utilizes a well-established reference dataset from one medical institution to build these models and employs a form of *transfer learning* to apply the models to data collected at other medical institutions. The overall problem setting is explained [here](framework_introduction.ipynb). 

## Technical details of the adaptation moddule

For domain adaptation with a limited number of samples, a new algorithm called `bpca_impute` has been developed. This module implements a data imputation approach using Bayesian probabilistic principal component analysis. One significant advantage of our BPCA-based imputation is that it is essentially parameter-free. In `impute_bpca_ard`, the dimensionality of the latent principal subspace, which is the critical parameter in any PCA-based algorithm, is automatically determined through an automatic relevance determination (ARD) mechanism. This feature makes it a preferred choice when dealing with a limited number of samples.

For more technical details, please refer to the notebooks:

For the technical detail, see the notebooks:
- [English](https://github.com/Idesan/bpca/blob/main/bmpca_impute.ipynb)
- [Japanese](https://github.com/Idesan/bpca/blob/main/bmpca_impute_JPN.ipynb)
