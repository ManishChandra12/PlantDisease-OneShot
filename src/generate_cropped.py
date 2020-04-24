import os
import shutil
import pandas as pd
import cv2
import split_folders


def main():
	if os.path.exists('cropped'):
		shutil.rmtree('cropped')

	os.mkdir('cropped')
	for subdir, dirs, files in os.walk('PlantDoc-Dataset'):
		os.mkdir(os.path.join('cropped', subdir))

	for fol in ['train', 'test']:
		df = pd.read_csv(fol + '_labels.csv')
		for subdir, dirs, files in os.walk('PlantDoc-Dataset/' + fol):
			for file in files:
				filepath = os.path.join(subdir, file)
				if filepath.endswith('.jpg'):
					img = cv2.imread(filepath)
					relevant = df[df['filename'] == file]
					xmin = relevant['xmin'].values
					xmax = relevant['xmax'].values
					ymin = relevant['ymin'].values
					ymax = relevant['ymax'].values
					for i in range(len(xmin)):
						cropped = img[ymin[i]-1:ymax[i]-1, xmin[i]-1:xmax[i]-1,:]
						try:
							cv2.imwrite(os.path.join('cropped', filepath[:-4] + str(i) + '.jpg'), cropped)
						except:
							print(filepath)
							print(img.shape)
							print(xmin[i]-1, xmax[i]-1, ymin[i]-1, ymax[i]-1)

	for subdir, dirs, files in os.walk('PlantDoc-Dataset/test'):
		for file in files:
			filepath = os.path.join(subdir, file)
			if filepath.endswith('.jpg'):
				shutil.move(filepath, filepath.replace('test', 'train'))

	shutil.rmtree('cropped/PlantDoc-Dataset/test')
	shutil.move('cropped/PlantDoc-Dataset/train', 'cropped/PlantDoc-Dataset/all')
	split_folders.ratio('cropped/PlantDoc-Dataset/all', output='cropped/PlantDoc-Dataset/splitted', seed=1337, ratio=(0.6, 0.2, 0.2))
	shutil.rmtree('cropped/PlantDoc-Dataset/all')
	shutil.move('cropped/PlantDoc-Dataset/splitted/val', 'cropped/PlantDoc-Dataset/val')
	shutil.move('cropped/PlantDoc-Dataset/splitted/train', 'cropped/PlantDoc-Dataset/train')
	shutil.move('cropped/PlantDoc-Dataset/splitted/test', 'cropped/PlantDoc-Dataset/test')
	shutil.rmtree('cropped/PlantDoc-Dataset/splitted')
	shutil.rmtree('cropped/PlantDoc-Dataset/train/Tomato two spotted spider mites leaf')
	shutil.rmtree('cropped/PlantDoc-Dataset/test/Tomato two spotted spider mites leaf')
	shutil.rmtree('cropped/PlantDoc-Dataset/val/Tomato two spotted spider mites leaf')

	for subdir, dirs, files in os.walk('cropped/PlantDoc-Dataset/test'):
		for file in files:
			filepath = os.path.join(subdir, file)
			if filepath.endswith('.jpg'):
				shutil.move(filepath, filepath.replace('test', 'train'))
	shutil.rmtree('PlantDoc-Dataset/test')
	shutil.move('PlantDoc-Dataset/train', 'PlantDoc-Dataset/all')
	split_folders.ratio('PlantDoc-Dataset/all', output='PlantDoc-Dataset/splitted', seed=1337, ratio=(0.6, 0.2, 0.2))
	shutil.rmtree('PlantDoc-Dataset/all')
	shutil.move('PlantDoc-Dataset/splitted/val', 'PlantDoc-Dataset/val')
	shutil.move('PlantDoc-Dataset/splitted/train', 'PlantDoc-Dataset/train')
	shutil.move('PlantDoc-Dataset/splitted/test', 'PlantDoc-Dataset/test')
	shutil.rmtree('PlantDoc-Dataset/splitted')
	shutil.rmtree('PlantDoc-Dataset/train/Tomato two spotted spider mites leaf')
	shutil.rmtree('PlantDoc-Dataset/test/Tomato two spotted spider mites leaf')
	shutil.rmtree('PlantDoc-Dataset/val/Tomato two spotted spider mites leaf')


if __name__ == '__main__':
	main()
