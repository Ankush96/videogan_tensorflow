import os
import glob
import sys
from skimage import io
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import argparse

REPLACE = False
PATH = './samples'

def build_gif(imgs, fname):
 
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
 
	ims = map(lambda x: (ax.imshow(x), ax.set_title('')), imgs)
 
	im_ani = animation.ArtistAnimation(fig, ims, interval=33, repeat_delay=0, blit=False)
 
	im_ani.save(fname[:-3]+'gif', writer='imagemagick')
 
	if REPLACE:
		os.remove(fname)
 
	return


def convert(files):
	for i,f in enumerate(files):
		img = io.imread(f)
	 	img = img.reshape((32,64,64,3))
	 	img = [a for a in img]

	 	sys.stdout.write("\rProcessing file %i/%i"%(i+1,len(files)))
		sys.stdout.flush()

	 	build_gif(img, f)
	print('\n')
	return


def main():

	global REPLACE, PATH

	parser = argparse.ArgumentParser()
	parser.add_argument("--replace", help="If true, deletes the original jpg files",
						type=bool, default=False)
	parser.add_argument("--path", help="Root directory of samples",
						type=str, default='./samples')
	args = parser.parse_args()

	REPLACE = args.replace
	PATH =args.path
	
	data = []
	pattern = "*.jpg"
	for dr,_,_ in os.walk(PATH):
		data.extend(glob.glob(os.path.join(dr,pattern)))

	if len(data) > 0:
		convert(data)
 	

if __name__ == '__main__':
	main()