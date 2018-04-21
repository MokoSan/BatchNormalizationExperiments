#!/bin/bash

echo "~ C1 Secret ~"

echo "Creating Virtual Env"
virtualenv -v env

echo "Activating Virtual Env"
source ./env/bin/activate

echo "Installing Requirements"
pip install -r requirements.txt
