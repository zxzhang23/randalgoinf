# randalgoinf: Statistical Inference via Randomized Algorithms

## Overview

This package implements various statistical inference methods for the deterministic target of a sequence of randomized algorithms as developed in Zhang et al. (2023). The general methods---including sub-randomization, multi-run plug-in, and multi-run aggregation---require the multiple runs of the randomized algorithm. The framework's applicability is demonstrated through inference for least squares parameters via sketch-and-solve, partial sketching, and iterative Hessian sketching. These methods are also applied in stochastic optimization contexts, including SGD with Polyak-Ruppert averaging and momentum-based approaches. The package also incorporates classical pivotal inference methods, supported by newly developed theoretical results for sketched least squares estimators.

## Installation

You can install the development version from GitHub:

```r
# Install devtools if you haven't already
install.packages("devtools")

# Install randalgoinf
devtools::install_github("zxzhang23/randalgoinf")
```

## References

Zhang, Z. et al. (2023). A Framework for Statistical Inference via Randomized Algorithms. arXiv preprint. Available at: https://arxiv.org/pdf/2307.11255

