# Knowledge Distillation Through Time Enables Hierarchial Predictive Coding #

## Requirements ## 

Inidividual module requirements for the project can be found within the parent file "requirements.txt". The version of Python used was 3.11.7. The verson of CUDA used was 12.1.

## Dataset Description ##

This is the CHBMIT dataset. This dataset consists of 23 patients taken from the Childrens Hospital Boston. Each patient has intercranial EEG recordings varying in length, the amount of seizures, and channels. The full dataset is approximately 50GB, and can be downloaded at https://physionet.org/content/chbmit/1.0.0/. Once the dataset has been downloaded, please place each of the patient data folders within the "Dataset" folder. Although there are 23 patients in total, we train only on a select few which have enough preictal data to train on, these can be found within "main.py". 

## Code Execution ##

This folder contains three main files for execution. "seizure_prediction.py", "seizure_detection.py", and "knowledge_distillation.py". The latter two files perform seizure prediction and detection using CNNLSTMS, which are in the models folder. It is best practice to run the "seizure_detection.py" file and save the seizure detector models, as they will be called during the forward pass in knowledge distillation. As for the main file here, "knowledge_ditillation.py", it contains a few key parameters. Namely the patient and the value of alpha. The patient is simply which patient we choose to perform FGL on, and alpha tunes the loss of CE and KL Divergence. 


## Parameter Tuning ## 

As mentioned in our paper, we use two main parameters in FGL for the distillation process, alpha and temperature. 
* Alpha: scales the cross-entroy loss
* Temperature: scales the softmax for KL loss. 

## Preprocessing Details ##

Finer details of the project, including the preprocessing code, can be found within the "utils" folder. This folder contains the preprocessing for both the teacher and student models. In summary, we first classify the data as either preictal, ictal or interical. Then, we take short time fourier transforms at a sampling rate of 256hz, and convert this into a numpy array. For the student model, we use a seizure occurence period of 30 minutes, and a seizure prediction horizon of 5 minutes. 