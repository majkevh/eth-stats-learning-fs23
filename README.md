# Statistical learning theory
## Markov Chain Monte Carlo Sampling
<img align="right" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/tsp.png"></img>
<img align="right" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/mcmc.png"></img>
This project delves into various MCMC sampling techniques, which are detailed in the paper of [Andrieu, Freitas, Doucet, Jordan (2003)](https://www.cs.princeton.edu/courses/archive/spr06/cos598C/papers/AndrieuFreitasDoucetJordan2003.pdf). These methods are applied to two distinct problem domains: (1) image denoising with  underlying Ising suumption,  (2) the Traveling Salesman Problem.
<br/><br/>
## Deterministic Annealing
<img align="left" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/bif.png"></img>
<img align="left" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/diag.png"></img>
Implementation  of the Deterministic Annealing (DA) algorithm as outlined in the paper titled ["Deterministic annealing for clustering, compression, classification, regression, and related optimization problems"](https://ieeexplore.ieee.org/document/726788) is pursued, followed by an empirical examination of its phase transition characteristics.
<br/><br/>
## Histogram Clustering
<img align="right" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/color.png"></img>
In this coding exercise, the objective is to perform image segmentation using Histogram Clustering (HC). The implementation will encompass two distinct approaches: Maximum a Posterior Probability (MAP) and Deterministic Annealing (DA) for predicting the cluster membership of individual pixels as outlined in ["Histogram clustering for unsupervised image segmentation"](http://ieeexplore.ieee.org/document/784981).
## Constant Shift Embedding
<img align="left" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/cse.png"></img>
This project delves into a method known as Constant Shift Embedding, which facilitates the embedding of pairwise clustering problems within vector spaces while maintaining the integrity of the cluster structure. The coding assignment consists in the implementation of the algorithm described in ["Optimal cluster preserving embedding of nonmetric proximity data"](https://ieeexplore.ieee.org/document/1251147) to cluster the groups of research community members based on the email correspondence matrix.
<br/><br/>
## Mean Field Approximation
<img align="right" height="110" src="https://github.com/majkevh/eth-stats-learning-fs23/blob/main/img/mfa.png"></img>
This code employs the methodology of Mean Field Approximation (MFA) as described in the publication ["An Introduction to Variational Methods for Graphical Models"](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf) applied to two distinct problems: (1) The 2D Ising model (2) A Wine Dataset.