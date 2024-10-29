#!/bin/bash

if [ ! -d ./pretrained_models ]; then
  mkdir ./pretrained_models
fi
wget -O pretrained_models.zip "https://nas-graphics.unizar.es/s/2ZDbY67oTyJLPKo/download/pretrained_models.zip" 
unzip pretrained_models.zip -d ./pretrained_models
