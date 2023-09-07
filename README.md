# bpca

A collection of Bayesian principal component analysis and related algorithms.

## bpca_impute
This module implements a data imputation approach using probabilistic principal component analysis, which was proposed in

> Takumi Kitamoto, Tsuyoshi Idé, Yuta Tezuka, Norio Wada, Yui Shibayama, Yuya Tsurutani, Tomoko Takiguchi, Sachiko Suematsu, Kei Omata, Yoshikiyo Ono, Ryo Morimoto, Yuto Yamazaki, Jun Saito, Hironobu Sasano, Fumitoshi Satoh, and Tetsuo Nishikawa, Identifying Primary Aldosteronism Patients who Require Adrenal Venous Sampling: A Multi-center Study, under review.

One main advantage of the BPCA-based imputation is that it is virtually parameter-free. The dimensionality of the latent space is a critical parameter in any PCA-based algorithm. In `bpca`, it can be automatically determined via an automatic relevance determination mechanism. This feature makes `bpca` a preferred choice when the number of samples is limited. 

For the technical detail, see the notebooks:
- [English](https://github.com/Idesan/bpca/blob/main/bmpca_impute.ipynb)
- [Japanese](https://github.com/Idesan/bpca/blob/main/bmpca_impute_JPN.ipynb)
