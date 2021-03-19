import os
import sys
import cv2
import time
import torch
import shutil
import subprocess
import numpy as np
from PIL import Image
from config import cfg
from model import ResNetUNet
from detect import resize_img, get_score, restore_bboxes


def validate_clockwise_points(points):
	"""
	Validates that the points that the 4 points that dlimite a polygon are in clockwise order.
	"""
	if len(points) != 8:
	    raise Exception("Points list not valid." + str(len(points)))
	
	point = [
	            [int(points[0]) , int(points[1])],
	            [int(points[2]) , int(points[3])],
	            [int(points[4]) , int(points[5])],
	            [int(points[6]) , int(points[7])]
	        ]
	edge = [
	            ( point[1][0] - point[0][0])*( point[1][1] + point[0][1]),
	            ( point[2][0] - point[1][0])*( point[2][1] + point[1][1]),
	            ( point[3][0] - point[2][0])*( point[3][1] + point[2][1]),
	            ( point[0][0] - point[3][0])*( point[0][1] + point[3][1])
	]
	
	summatory = edge[0] + edge[1] + edge[2] + edge[3];
	return False if summatory > 0 else True


def detect_score(model, device, dataset_path, score_path):
	img_files = os.listdir(dataset_path)
	img_files = sorted([os.path.join(dataset_path, img_file) for img_file in img_files])

	for i, img_file in enumerate(img_files):
		save_file = os.path.join(score_path, os.path.basename(img_file).split('.')[0].strip('ts_') + '.npy')
		res = {}

		img = Image.open(img_file).convert('RGB')
		print(img_file, img.height, img.width)
		res['origin_h'] = img.height
		res['origin_w'] = img.width
		img = resize_img(img, cfg.test.short_side)
		res['resize_h'] = img.height
		res['resize_w'] = img.width
		cls, rho, theta = get_score(img, model, device)
		res['cls'] = cls
		res['rho'] = rho
		res['theta'] = theta
		np.save(save_file, res)
		sys.stdout.flush()


def detect_txt(score_path, txt_path, prob_thresh):
	if os.path.exists(txt_path):
		shutil.rmtree(txt_path)
	os.mkdir(txt_path)

	assert(os.path.exists(score_path))
	for i, npy_file in enumerate(os.listdir(score_path)):
		print(npy_file)
		res = np.load(os.path.join(score_path, npy_file), allow_pickle=True).item()
		save_file = os.path.join(txt_path, 'res_' + os.path.basename(npy_file).split('.')[0] + '.txt')
		bboxes = restore_bboxes(res['cls'], res['rho'], res['theta'], prob_thresh)

		vertices = []
		conf = []
		for bbox in bboxes:
			pts = bbox[:8].reshape((4,2)) * 4 * (res['origin_w'] / res['resize_w'] + res['origin_h'] / res['resize_h']) / 2 
			vertices.append(pts)
			conf.append(bbox[8])

		with open(save_file, 'w') as f:
			for j, vertice in enumerate(vertices):
				pts = np.int32(np.around(vertice.flatten()))
				if not validate_clockwise_points(pts):
					continue
				pts = pts.astype(str).tolist()
				pts = ','.join(pts) + ',' + '0.9' + '\n' # str(conf[j])
				f.write(pts)
		sys.stdout.flush()


def vis(img_path, txt_path):
	assert(os.path.exists(img_path))
	assert(os.path.exists(txt_path))

	for i, txt_file in enumerate(os.listdir(txt_path)):
		print('evaluating {} image'.format(i), end='\r')
		img = np.array(Image.open('../data/ICDAR2017/test_img/' + txt_file.replace('txt', 'jpg')))

		with open(os.path.join(txt_path, txt_file), 'r') as f:
			lines = f.readlines()
		for line in lines:
			info = line.strip().split(',')
			assert(len(info) % 2 == 0)
			cords = np.array(list(map(int, info))).reshape((-1,2))
			pts = cords.reshape((-1,1,2)).astype(np.int32)
			cv2.polylines(img,[pts],True,(0,215,255),3)

		Image.fromarray(img).save('./vis/' + txt_file.replace('txt', 'png'))



if __name__ == '__main__': 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ResNetUNet(backbone='50')
	model.to(device)
	model.eval()

	score_path = './score'
	txt_path = './submit'

	max_fscore, best_epoch, best_thresh = 0, 0, 0
	st = time.time()
	for i in range(150, 151):
		model_pth = './pths/train/model_epoch_{}.pth'.format(i)
		model.load_state_dict(torch.load(model_pth))

		if os.path.exists(score_path):
			shutil.rmtree(score_path)
		os.mkdir(score_path)

		print('model processing...')
		print(model_pth)
		st_d = time.time()
		detect_score(model, device, cfg.test.dataset_test, score_path)
		print('gpu time is {}'.format(time.time()-st_d))

		print('post processing...')
		for prob_thresh in np.arange(0.8, 0.85, 0.1):
			if os.path.exists(txt_path):
				shutil.rmtree(txt_path)
			os.mkdir(txt_path)

			print(prob_thresh)
			detect_txt(score_path, txt_path, prob_thresh)

			os.chdir(txt_path)
			res = subprocess.getoutput('zip -q submit_{}_{}.zip *.txt'.format(i, prob_thresh))
			res = subprocess.getoutput('mv submit_{}_{}.zip ../'.format(i, prob_thresh))
			os.chdir('../')

	print('time is {}'.format(time.time() - st))
