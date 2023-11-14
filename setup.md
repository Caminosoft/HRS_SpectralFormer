# Setup Commands to run the code.

## Introduction
Below Commands will help us to setup the environment, install the requirements and make the code running.

## Clone the Repository

Clone the repository using git clone {http link}

## Environment Setup
Detail the steps needed to set up the Python environment required to run the code. Include guidance for creating a virtual environment, if necessary.

Create the Environment. 

`python3 -m venv pytorch_env`

Activate the environment. 

`source pytorch_env/bin/activate`


## Requirements
List the dependencies needed to run the code. Instructions on how to install them via `requirements.txt` or any other package management tool.

`pip install -r requirements.txt`


## How to Run


To Run the Model Training function with the provide custom data:

`python demo.py --dataset='Custom' --flag_test='train' --epoches=10 --patches=3 --band_patches=3 --mode='CAF' --weight_decay=0`

Note:

Change the parameters as per the requirements.

To Run the Inference function with the provided custom data:

`python demo.py --dataset='Custom' --flag_test='inference' --epoches=10 --patches=3 --band_patches=3 --mode='CAF' --weight_decay=0`