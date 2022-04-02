import os
import cv2

dirnum = "0000/"
indir = "/home/senior/workspace/data/video/"+ dirnum
outdir = "/home/senior/workspace/data/picture/"+ dirnum


os.makedirs(outdir, exist_ok=True)
names = sorted(os.listdir(indir))
videolst = [f'{os.path.join(indir, name)}' for name in names]
print(videolst)

for videopath in videolst:
	vidcap = cv2.VideoCapture(videopath)

	while(vidcap.isOpened()):
		ret, img = vidcap.read()
		if ret:
			num = str(int(vidcap.get(1))).zfill(5)
			outpath = outdir + f'frame{num}.jpg'
			print(outpath)
			cv2.imwrite(outpath, img)
	vidcap.release()