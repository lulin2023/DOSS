# DOSS Repository Overview
This repository contains the codes to reproduce the experiments results in *Diversified Online Sample Selection via Predictive Inference (2023)*.

## Introduction of DOSS
Real-time decision making gets more attention in the big data era. Here, we consider the problem of sample selection in the online setting, where
one encounters a possibly infinite sequence of individuals collected by time with covariate information available. The goal is to obtain a given number of informative individuals that are characterized by their unobserved responses. We derive a new algorithm named as DOSS that is able to control the online false selection rate (FSR) and real-time expected similarity of the selected samples simultaneously, allowing us to find more preferable individuals that exhibit certain diversity. The key elements are to quantify the uncertainty of response predictions via predictive inference and to characterize the proportion of false selections and diversity in a sequential manner.

## Folder contents

- **R**: Function definitions
- **man**: Package documentation files
- **Experiments codes**: The codes to reproduce the experiments results in the paper.
- **data set**: Two real data sets in application.

## Installing the DOSS package

We can use `devtools` to install the `DOSS` package.

```
# Install and load devtools
install.packages("devtools")
library(devtools)

# Install and load DOSS
devtools::install_github("lulin2023/DOSS")
library(DOSS)
```

