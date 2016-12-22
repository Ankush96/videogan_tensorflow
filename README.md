# videogan_tensorflow
Implementation of videogan in tensorflow

## Setup

You will need Python 3.5 for this code. If you do not already have it, make an environment.
```sh
conda create -n tf python=3.5
source activate tf
```
Here tf is the name of the envionment. <br>
Install numpy, scikit-image, matplotlib, scipy and tensorflow in this environment using `pip install library_name` <br>

Make a folder named data in this repository and put all images (the jpeg files as found [here](http://web.mit.edu/vondrick/tinyvideo/)) inside the data folder. You can put it any subdirectory you want. Due to lack of hard disk space, I could only test the code for a few images. <br>

All the hyperparameters are saved in the options.py file. If you are running the code for the first time, uncomment the lines 27-36 in utils/\_\_init\_\_.py. It will test if the images are corrupted or not and prompt you to delete those images and also save the mean image. After deleting the files prompted, comment the lines and run again.

To run the script
```sh
python main.py
```
