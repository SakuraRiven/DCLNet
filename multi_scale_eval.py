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
import lanms


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


def detect_score(model, device, dataset_path, score_path, short_side):
	img_files = os.listdir(dataset_path)
	img_files = sorted([os.path.join(dataset_path, img_file) for img_file in img_files])

	for i, img_file in enumerate(img_files):
		save_file = os.path.join(score_path, '{}_'.format(short_side) + os.path.basename(img_file).split('.')[0].strip('ts_') + '.npy')
		res = {}

		img = Image.open(img_file).convert('RGB')
		print(short_side, img_file)
		res['origin_h'] = img.height
		res['origin_w'] = img.width
		img = resize_img(img, short_side)
		res['resize_h'] = img.height
		res['resize_w'] = img.width
		cls, rho, theta = get_score(img, model, device)
		res['cls'] = cls
		res['rho'] = rho
		res['theta'] = theta
		np.save(save_file, res)
		sys.stdout.flush()


def detect_txt(score_path, txt_path, prob_thresh, short_side):
	npy_list = []
	for score in os.listdir(score_path):
		npy_list.append(score.split('_',1)[1].split('.')[0])
	npy_list = list(set(npy_list))

	for npy_file in sorted(npy_list):
		print(npy_file)
		save_file = os.path.join(txt_path, 'res_' + npy_file + '.txt')
		vertices = []
		for short_side in short_sides:
			res = np.load(os.path.join(score_path, '{}_'.format(short_side) + npy_file + '.npy'), allow_pickle=True).item()
			bboxes = restore_bboxes(res['cls'], res['rho'], res['theta'], prob_thresh, nms=False)

			for bbox in bboxes:
				pts = bbox[:8] * 4 * (res['origin_w'] / res['resize_w'] + res['origin_h'] / res['resize_h']) / 2 
				vertices.append(pts.tolist() + [bbox[8]])

		bboxes = np.array(vertices)
		bboxes = lanms.merge_quadrangle_n9(bboxes.astype('float32'), cfg.test.nms_thresh)

		with open(save_file, 'w') as f:
			for vertice in bboxes:
				pts = np.int32(np.around(vertice[:8].flatten()))
				if not validate_clockwise_points(pts):
					continue
				pts = pts.astype(str).tolist()
				pts = ','.join(pts) + ',' + '0.9' + '\n' # str(conf[j])
				f.write(pts)
		sys.stdout.flush()



if __name__ == '__main__': 
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model = ResNetUNet(backbone='50')
	model.to(device)
	model.eval()

	score_path = './ms_score'
	txt_path = './ms_submit'

	st = time.time()
	i = 150
	model_pth = './pths/train/model_epoch_{}.pth'.format(i)
	model.load_state_dict(torch.load(model_pth))
	short_sides = [640, 1920] #list(range(640, 1921, 80)) 
	if os.path.exists(score_path):
		shutil.rmtree(score_path)
	os.mkdir(score_path)

	print('model processing...')
	for short_side in short_sides:
		st_d = time.time()
		detect_score(model, device, cfg.test.dataset_test, score_path, short_side)
		print('gpu time is {}'.format(time.time()-st_d))

	prob_thresh = 0.8
	if os.path.exists(txt_path):
		shutil.rmtree(txt_path)
	os.mkdir(txt_path)

	print(prob_thresh)
	detect_txt(score_path, txt_path, prob_thresh, short_sides)

	os.chdir(txt_path)
	res = subprocess.getoutput('zip -q ms_submit_{}_{}.zip *.txt'.format(i, prob_thresh))
	res = subprocess.getoutput('mv ms_submit_{}_{}.zip ../'.format(i, prob_thresh))
	os.chdir('../')

	print('time is {}'.format(time.time() - st))
