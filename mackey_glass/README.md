# Knowledge Distillation Through Time Enables Hierarchial Predictive Coding #

## Requirements ## 

Inidividual module requirements for the project can be found within the parent file "requirements.txt". The version of Python used was 3.11.7. The verson of CUDA used was 12.1.

## Dataset Description ##

This is the Mackey Glass dataset. Unlike the previous two seizure datasets, this is a (decievingly) simpler dataset that consists of samples extracted from the Makcey Glass differntial equation. 

## Code Execution ##

This folder contains one main files for execution, "KDTT.py." This file with run all three models, and print out losses as well. Details of training are well detailed within the files and should be easy to navigate.

## Parameter Tuning ## 

As mentioned in our paper, we use two main parameters in KDTT for the distillation process, alpha and temperature. 
* Alpha: scales the cross-entroy loss
* Temperature: scales the softmax for KL loss. 

## Preprocessing Details ##

The "utils.py" file contains the preprocessing for this dataset. It simply generates values, places the targets into bins, and returns the data loaders that we use in the main file. a visualization of the binning process has been included for easy of viewing, that can be uncommented. 