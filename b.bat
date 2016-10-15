#! /bin/bash

cd ~
mkdir Documents
cd Documents
wget "https://github.com/srhumir/kaggle-regression/raw/master/loss_train.csv"
wget "https://github.com/srhumir/kaggle-regression/raw/master/X_train.csv"
wget "https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh"
bash Anaconda3-4.2.0-Linux-x86_64.sh
conda uninstall scikit-learn
pip install sklearn
pip install Theano
pip install keras
cd ~
mkdir .keras
cd .keras
wget "https://github.com/srhumir/kaggle-regression/raw/master/keras.json"
cd ~
cd Documents
jupyter console
