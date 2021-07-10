#!/bin/bash
cd ~/workspace/slvideorepr/slvideorepr
source ../.env/bin/activate
ipython youscrap.py
sleep 25
ipython youscrap.py
deactivate
