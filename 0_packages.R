#!/usr/bin/env Rscript
# ------------------------------------------------------------------------------
# Filename:    0.packages.R
# Description: Installs the required R packages
# Duration:    ~ 30 mins
# ------------------------------------------------------------------------------

pkg <- c("ca", "data.table", "rstan")
new_pkg <- pkg[!(pkg %in% installed.packages()[,"Package"])]
if(length(new_pkg)) install.packages(new_pkg, repos = "http://cran.us.r-project.org")
