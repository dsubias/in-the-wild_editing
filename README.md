# FaderNetworks-Materials


## Organization of the code

* `agents.lightningModule` : The main training file (architecture, optimisation scheme) is in `agents.lightningModule`. This file contains a Pytorch lighting abstraction to train all models
. All of the optimization/training procedure is in `FaderNetPL`, the param `network` in the config file determinates the model to use:
  * `FaderNet` : architecture of $G_1$ without the normals, contains most of the code (equivalent to faderNet)
  * `FaderNetWithNormals` : $G_1$
  * `FaderNetWithNormals2Steps` : $G_2$
* `configs` : configuration files to launch the trainings or test
* `datasets` : code to read the datasets
* `experiments` : snapshots of experiments
* `models` : code of the networks
* `utils` : various utilities
* `train.py` : Its contains the code to train the model
  * `--config`: The path of configuration file in yaml format
