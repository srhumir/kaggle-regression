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
sudo yum install gcc-c++
sudo yum install git
sudo rm Anaconda3-4.2.0-Linux-x86_64.sh
git init
git remote add origin https://github.com/srhumir/aws2
git pull https://github.com/srhumir/aws2
git config user.email "srh.umir@gmail.com"
git config user.name "Reza"
git add -A
git commit -m "initiate"
git push --set-upstream origin master
git config --global credential.helper 'cache --timeout 72000'
git config --global push.default simple
