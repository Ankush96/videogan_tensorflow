Generating Videos with Scene Dynamics
--------------------------------------
This repository contains an implementation of [Generating Videos with Scene Dynamics](http://web.mit.edu/vondrick/tinyvideo/) by Carl Vondrick, Hamed Pirsiavash, Antonio Torralba. The model learns to generate tiny videos using adversarial networks.

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
python main.py
```
