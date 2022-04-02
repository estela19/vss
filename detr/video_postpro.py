import cv2
import os

pathIn= '/home/senior/workspace/output/0000/'
pathOut = '/home/senoir/workspace/output/0000.MP4'
fps = 30
frame_array = []

paths = sorted(os.listdir(pathIn))
for idx , path in enumerate(f'{os.path.join(pathIn,path)}' for path in paths) : 
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