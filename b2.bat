#! /bin/bash

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
wget "https://github.com/srhumir/kaggle-regression/raw/master/boost_for_aws.py" 

