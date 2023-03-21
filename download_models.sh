#!/bin/bash

if [ ! -d ./pretrained_models ]; then
  mkdir ./pretrained_models
fi
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1pXggdRjXVKgKF96ax2xeC-Yw4s5DxgAv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1pXggdRjXVKgKF96ax2xeC-Yw4s5DxgAv" -O pretrained_models.zip && rm -rf /tmp/cookies.txt
unzip pretrained_models.zip -d ./pretrained_models