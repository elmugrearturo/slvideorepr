#!/bin/bash
cd ~/workspace/slvideorepr/slvideorepr
source ../.env/bin/activate
ipython download_from_list.py ../bin/lsc.wikisign.org.bin
sleep 15
ipython download_from_list.py ../bin/lsc.wikisign.org.bin
sleep 15
ipython download_from_list.py ../bin/lsc.wikisign.org.bin
sleep 15
ipython download_from_list.py ../bin/lsc.wikisign.org.bin
deactivate
