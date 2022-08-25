# Dynamic Lexicon

This repo contains the scripts to reproduce experiments described in the following paper :

Temporão, Mickael, et al. "Ideological scaling of social media users: a dynamic lexicon approach." Political Analysis 26.4 (2018): 457-473.

Code is written in python 2 (needs to be upgraded to python 3) and R. 
Pipfile specifies the python virtual environment.

## Short description of the files

- params_grid.csv : specification of hyperparameters

### 0 - Packages, Data sample and Anonymization
- 0_packages.R : install the packages required for the R scripts
- 0_resample.py : generate the Twitter sample used for the study from VPL panel
- 0b_anonymization.R : anonymize VPL data

### 1 - Network ideologies
- 1_network_matrix.py : create the adjacency graph (matrix structure)
- 2_network_ideologies.R : scale ideologies from adjacency graph

### 2 - Text
- 3_textual_ideologies : scale ideologies from tweets

### 3 - Appendix
- 4_appendix_filters : perform extra analysis on the effect of filters
- 5_aggregate_features : different scalings obtained from different methods altogether
- 6_learn_features : try to predict vote intention using scaling features
- 7_generate_figures : generate figures used in the paper
- appendix.py : different utils functions for reviewers comments
- models.py : utils imported for machine learning use

