Silicon Wafer Defect Detection Using Vision Transformers (WM-811K)

This repository implements a robust 
training pipeline for silicon wafer defect classification using a Vision
 Transformer (ViT-Tiny) model trained on the WM-811K / LSWMD dataset. 

The training script supports 
automatic checkpointing, safe interruption handling, and true 
pause/resume functionality without loss of progress.

Project Structure

The repository contains only source code and configuration files. Large 
files such as datasets, virtual environments, and trained models are 
intentionally excluded.

Key files:

silicondefect_transformer_train.py – main training script
.gitignore – excludes datasets, virtual environments, and model files
requirements.txt – Python dependencies
README.md – project documentation

Dataset

The project uses the WM-811K (LSWMD) silicon wafer map dataset, 
available publicly on Kaggle. Due to size and licensing constraints, the
 dataset is not included in the repository.


Dataset source:

https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map

After downloading, the dataset must be placed at:

dataset/LSWMD.pkl

The dataset path can be modified in the training script if required.


Environment Setup

A Python virtual environment is recommended to ensure dependency isolation.


Create a virtual environment:

python -m venv wm811k_env


Activate the environment (Windows PowerShell):

wm811k_env\Scripts\activate


Install required dependencies:

pip install -r requirements.txt


Model Training

Training is initiated by running the main script:
python silicondefect_transformer_train.py


During execution, the script:

loads and preprocesses the LSWMD dataset
performs label cleaning and encoding
resizes wafer maps to 128×128 resolution
trains a ViT-Tiny (patch size 16) classification model
evaluates validation accuracy after each epoch


The input resolution is reduced 
from the standard 224×224 to 128×128 to improve performance on laptop 
hardware without significant accuracy degradation.


Checkpointing and Resume

The training process automatically saves a checkpoint at the end of 
every completed epoch. The checkpoint includes the model state, 
optimizer state, and training metadata.


Checkpoint file:

vit_wafer_checkpoint.pth


If training is interrupted or 
stopped, rerunning the script will automatically resume training from 
the last completed epoch without manual intervention.


Pause and Resume Without Stopping

The training process can be paused without terminating the Python 
process. If a file named PAUSE.txt exists in the project directory, 
training will pause automatically. Deleting this file resumes training 
from the next batch. This mechanism allows safe temporary pauses during 
long training runs.


Final Model Output

Upon successful completion of all training epochs, the final trained model is saved as:

vit_wafer_final.pth


This file contains the trained 
weights along with the associated class label mapping and configuration 
parameters after 5 epochs, and can be used directly for inference or evaluation.

Important Notes

Virtual environments, dataset files, and trained model artifacts are 
excluded from version control via .gitignore. Users must download the 
dataset separately and recreate the virtual environment locally before 
running the code.
