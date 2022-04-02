import cv2
import os
from PIL import Image
import imageio

pathIn= '/home/senior/workspace/output/0000/'
pathOut = '/home/senoir/workspace/output/test.gif'
fps = 30

def make_video():
	frame_array = []

	paths = sorted(os.listdir(pathIn))
	for idx , path in enumerate(f'{os.path.join(pathIn,path)}' for path in paths): 
		img = cv2.imread(path)
		height, width, layers = img.shape
		size = (width,height)
		frame_array.append(img)

	out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc('M','J','P','G'), fps, size)
	#breakpoint()
	if not out.isOpened():
		print("Fileopen failed")

	for i in range(len(frame_array)):
		# writing to a image array
		out.write(frame_array[i])
	out.release()

def make_gif():
	imgs = []
	paths = sorted(os.listdir(pathIn))
	for path in (f'{os.path.join(pathIn,path)}' for path in paths): 
		imgs.append(Image.open(path))
	imageio.mimsave(pathOut, imgs, fps=fps)

def make_gif2():
	anim_file = os.getcwd()+'/test.gif'

	with imageio.get_writer(anim_file, mode='I') as writer:
		names = sorted(os.listdir(pathIn))
		filenames = [f'{os.path.join(pathIn, name)}' for name in names]
		last = -1
		for i, filename in enumerate(filenames):
			frame = 2*(i**0.5)
			if round(frame) > round(last):
				last = frame
			else:
				continue
			image = imageio.imread(filename)
			writer.append_data(image)
		image = imageio.imread(filename)
		writer.append_data(image)

if __name__ == '__main__':
	make_gif2()