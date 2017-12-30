Generating Videos with Scene Dynamics
--------------------------------------
This repository contains an implementation of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. The model learns to generate tiny videos using adversarial networks.

Visualizations
---------------------------------
When trained on golf dataset, the following samples are obtained. <br>
<img src="https://github.com/Ankush96/videogan_tensorflow/blob/master/3.gif?raw=True" width="300"> 
<img src="https://github.com/Ankush96/videogan_tensorflow/blob/master/2.gif?raw=True" width="300"> <br>

Hopefully some careful parameter tuning will give better results



Setup
---------------------------------

You will need Python 3.5 for this code. If you do not already have it, make an environment.
```sh
conda create -n tf python=3.5
source activate tf
```
Here tf is the name of the envionment. <br>
Install numpy, scikit-image, matplotlib, scipy and tensorflow in this environment using `pip install library_name` <br>

Make a folder named data in this repository and put all images (the jpeg files as found [here](http://web.mit.edu/vondrick/tinyvideo/)) inside the data folder. You can put it any subdirectory you want. Due to lack of hard disk space, I could only test the code for a few images. <br>

Training
----------------------------------------
```sh
python main.py [Options]

Options:

# Data
  --dataset, './data', Path to data.
  --data_list, './data/list.pkl', Cached list of data.
  --mean_path, './data/mean.png', Path to mean of data.
  --sample_path, './samples', Path to save samples in.

# Training
  --batch_size, 20, Number of videos to process in a batch.
  --sample_size, 20, Number of videos to sample and save in a batch.
  --train_epochs, 10**10, Number of training epochs.
  --lrate_d, 1e-5, Learning rate for discriminator.
  --lrate_g, 1e-4, Learning rate for generator.
  --beta1_d, 0.5, beta1 for discriminator.
  --beta1_g, 0.5, beta1 for generator.

# Architecture
  --z_dim, 100, Dimension of initial noise vector.
  --gf_dim, 64, Conv kernel size of G.
  --df_dim, 64, Conv kernel size of D.
  --c_dim, 3, Number of input channels.
  --mask_penalty, 0.1, Lambda for L1 regularizer of mask.

# Model saving
  --checkpoint_dir, './checkpoints', Path to checkpoint models.
  --checkpoint_time, 30*60, Time to save checkpoints after. 
  --sampler_time, 30*60, Time to save samples in.
  --print_time, 60, Time to print loss.
  --calc_mean, False, Whether or not to calculate mean.


```
