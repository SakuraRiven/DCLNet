import cv2
import os
import math
import numpy as np
import time
import torch
import torchvision.transforms as transforms
import scipy.io as scio
from PIL import Image
from torch.utils import data
from config import cfg


def resize_img(img, vertices, min_side, max_side):
	resize_w = int(np.random.rand() * (max_side - min_side)) + min_side
	resize_h = int(np.random.rand() * (max_side - min_side)) + min_side
	w, h = img.size

	img = img.resize((resize_w, resize_h), Image.BILINEAR)

	for i, vertice in enumerate(vertices):
		for j in range(len(vertice)):
			if j % 2 != 0:
				vertice[j] *= resize_h / h
			else:
				vertice[j] *= resize_w / w
	return img


def crop_img(img, vertices, length, is_valid):
	# find random position
	remain_h = img.height - length
	remain_w = img.width - length
	start_w = int(np.random.rand() * remain_w)
	start_h = int(np.random.rand() * remain_h)
	box = (start_w, start_h, start_w + length, start_h + length)
	img_region = img.crop(box)

	for i, vertice in enumerate(vertices):
		for j in range(len(vertice)):
			if j % 2 != 0:
				vertice[j] -= start_h
			else:
				vertice[j] -= start_w
		if np.min(vertice) < 0 or np.max(vertice) > length:
			is_valid[i] = 0
	return img_region


def generate_label(img, vertices, is_valid, scale): # vertices is (c,r) not (x,y)
	label_height, label_width = int(np.around(img.height * scale)), int(np.around(img.width * scale))
	label_text, label_ignore, label_instance = [np.zeros((1, label_height, label_width)) for _ in range(3)]
	label_rho, label_theta = [np.zeros((4, label_height, label_width)) for _ in range(2)]
	for i, vertice in enumerate(vertices):
		temp = vertice*scale
		if is_valid[i] == 0:
			cv2.fillPoly(label_ignore[0,:,:], [temp.reshape((-1,2)).astype(np.int32)], 1)
			continue
		shrink_vertice = shrink_boundary(temp)
		cv2.fillPoly(label_text[0,:,:], [shrink_vertice.reshape((-1,2)).astype(np.int32)], 1)
		cv2.fillPoly(label_instance[0,:,:], [shrink_vertice.reshape((-1,2)).astype(np.int32)], i+1)

		r, c = np.where(label_instance[0,:,:] == i+1)
		for j in range(len(r)):
			label_rho[0,r[j],c[j]], label_theta[0,r[j],c[j]] = cal_rho_theta(temp[0]-c[j], r[j]-temp[1], temp[2]-c[j], r[j]-temp[3])  # c,r -> x,y
			label_rho[1,r[j],c[j]], label_theta[1,r[j],c[j]] = cal_rho_theta(temp[2]-c[j], r[j]-temp[3], temp[4]-c[j], r[j]-temp[5])  # c,r -> x,y
			label_rho[2,r[j],c[j]], label_theta[2,r[j],c[j]] = cal_rho_theta(temp[4]-c[j], r[j]-temp[5], temp[6]-c[j], r[j]-temp[7])  # c,r -> x,y
			label_rho[3,r[j],c[j]], label_theta[3,r[j],c[j]] = cal_rho_theta(temp[6]-c[j], r[j]-temp[7], temp[0]-c[j], r[j]-temp[1])  # c,r -> x,y
	return label_text, label_ignore, label_rho, label_theta 


def cal_rho_theta(x1, y1, x2, y2):
	# AX+BY+C=0
	A = y2-y1
	B = x1-x2
	C = x2*y1 - x1*y2
	rho = abs(C) / np.sqrt(A**2 + B**2 + 1e-8)

	vector = (x1-x2, y1-y2) if y1>=y2 else (x2-x1, y2-y1)
	cos_theta = vector[0] / np.sqrt(vector[0]**2 + vector[1]**2 + 1e-8)
	theta = np.arccos(cos_theta) # math.pi
	theta = theta - math.pi/2 if theta > math.pi/2 else theta + math.pi/2
	if B*C > 0: # 直线在原点下方
		theta += math.pi
	return rho, theta


def shrink_boundary(vertice, shrink_ratio=0.4):
	x1, y1, x2, y2, x3, y3, x4, y4 = vertice
	center_x = (x1 + x2 + x3 + x4) / 4
	center_y = (y1 + y2 + y3 + y4) / 4

	shrink_x1 = x1 + shrink_ratio*(center_x - x1)
	shrink_y1 = y1 + shrink_ratio*(center_y - y1)
	shrink_x2 = x2 + shrink_ratio*(center_x - x2)
	shrink_y2 = y2 + shrink_ratio*(center_y - y2)
	shrink_x3 = x3 + shrink_ratio*(center_x - x3)
	shrink_y3 = y3 + shrink_ratio*(center_y - y3)
	shrink_x4 = x4 + shrink_ratio*(center_x - x4)
	shrink_y4 = y4 + shrink_ratio*(center_y - y4)

	return np.array([shrink_x1, shrink_y1, shrink_x2, shrink_y2, shrink_x3, shrink_y3, shrink_x4, shrink_y4])


def merge_datasets(img_path_17, gt_path_17):
	total_img_files = []
	total_gt_files = []

	for img_17, gt_17 in zip(img_path_17, gt_path_17):
		total_img_files.extend([os.path.join(img_17, img_file) for img_file in sorted(os.listdir(img_17))])
		total_gt_files.extend([os.path.join(gt_17, gt_file) for gt_file in sorted(os.listdir(gt_17))])
	return total_img_files, total_gt_files


def extract_vertices_mlt(lines):
    vertices = []
    is_valid = []
    for line in lines:
        info = line.rstrip('\n').split(',', 9)
        assert(len(info) == 10)
        vertices.append(np.array(list(map(float, info[:8]))))
        if '#' in info[9]:
            is_valid.append(0)
        else:
            is_valid.append(1)
    return vertices, np.array(is_valid)


class FinetuneDataset(data.Dataset): # clockwise
	def __init__(self, cfg):
		super(FinetuneDataset, self).__init__()

		self.img_files, self.gt_files = merge_datasets(cfg.img_path, cfg.gt_path)

		self.scale = cfg.scale
		self.min_side = cfg.min_side
		self.max_side = cfg.max_side
		self.crop_length = cfg.crop_length
		self.t = transforms.Compose([transforms.ColorJitter(0.2, 0.2, 0.2, 0.2), \
                                     transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, index):
		img = Image.open(self.img_files[index]).convert('RGB')
		with open(self.gt_files[index], 'r') as f:
			lines = f.readlines()

		vertices, is_valid = extract_vertices_mlt(lines) # [ndarray:x1, y1, x2, y2..., ]
		img = resize_img(img, vertices, self.min_side, self.max_side)
		img = crop_img(img, vertices, self.crop_length, is_valid)

		label_text, label_ignore, label_rho, label_theta = generate_label(img, vertices, is_valid, self.scale)
		label_text, label_ignore, label_rho, label_theta = list(map(lambda x: torch.Tensor(x), [label_text, label_ignore, label_rho, label_theta]))

		return self.t(img), label_text, label_ignore, label_rho, label_theta


class SynthTextDataset(data.Dataset):
	def __init__(self, cfg):
		super(SynthTextDataset, self).__init__()
		self.img_path = cfg.img_path
		self.scale = cfg.scale
		self.min_side = cfg.min_side
		self.max_side = cfg.max_side
		self.crop_length = cfg.crop_length
		gt = scio.loadmat(os.path.join(cfg.gt_path, 'gt.mat'))
		self.img_files = gt['imnames'][0]
		self.wordBB    = gt['wordBB'][0]
		self.charBB    = gt['charBB'][0]
		self.txt       = gt['txt'][0]
		self.t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.mean, cfg.std)])

	def _wordBB2vertices(self, wordBB):
		vertices = []
		for i in range(wordBB.shape[-1]):
			vertices.append(wordBB[:,:,i].transpose().reshape(-1))
		return vertices

	def __len__(self):
		return len(self.img_files)

	def __getitem__(self, index):
		img = Image.open(os.path.join(self.img_path, self.img_files[index][0]))
		wordBB = self.wordBB[index].reshape((2,4,-1)) # 2x4xn
		vertices = self._wordBB2vertices(wordBB) # [ndarray(8,), ...]
		is_valid = np.ones(len(vertices))
		img = resize_img(img, vertices, self.min_side, self.max_side)
		img = crop_img(img, vertices, self.crop_length, is_valid)

		label_text, label_ignore, label_rho, label_theta = generate_label(img, vertices, is_valid, self.scale)
		label_text, label_ignore, label_rho, label_theta = list(map(lambda x: torch.Tensor(x), [label_text, label_ignore, label_rho, label_theta]))

		return self.t(img), label_text, label_ignore, label_rho, label_theta


if __name__ == '__main__':
	dataset = SynthTextDataset(cfg.pretrain)
	img, label_text, label_ignore, label_rho, label_theta = dataset[10]
	print(img.shape, label_text.shape, label_ignore.shape, label_rho.shape, label_theta.shape)
	img, label_text, label_ignore, label_rho, label_theta = [i.numpy() for i in [img, label_text, label_ignore, label_rho, label_theta]]
	img = img.transpose((1,2,0))
	img[:,:,0] = img[:,:,0] * 0.229 + 0.485
	img[:,:,1] = img[:,:,1] * 0.224 + 0.456
	img[:,:,2] = img[:,:,2] * 0.225 + 0.406
	Image.fromarray(np.uint8(img*255)).save('./img.bmp')

	if np.max(label_text) > 0:
		label_text = np.uint8((label_text[0,:,:] - np.min(label_text[0,:,:])) / np.max(label_text[0,:,:]) * 255)
	else:
		label_text = np.uint8(label_text[0,:,:]  * 255)
	Image.fromarray(label_text).save('./text.bmp')

	if np.max(label_ignore) > 0:
		label_ignore = np.uint8((label_ignore[0,:,:] - np.min(label_ignore[0,:,:])) / np.max(label_ignore[0,:,:]) * 255)
	else:
		label_ignore = np.uint8(label_ignore[0,:,:]  * 255)
	Image.fromarray(label_ignore).save('./ignore.bmp')

	if np.max(label_rho) > 0:
		label_rho = np.uint8((label_rho[0,:,:] - np.min(label_rho[0,:,:])) / np.max(label_rho[0,:,:]) * 255)
	else:
		label_rho = np.uint8(label_rho[0,:,:]  * 255)
	Image.fromarray(label_rho).save('./rho.bmp')
	print(label_theta.min(), label_theta.max())
	if np.max(label_theta) > 0:
		label_theta = np.uint8((label_theta[0,:,:] - np.min(label_theta[0,:,:])) / np.max(label_theta[0,:,:]) * 255)
	else:
		label_theta = np.uint8(label_theta[0,:,:]  * 255)
	Image.fromarray(label_theta).save('./theta.bmp')
	print(cal_rho_theta(0,1,1,0))
	print(cal_rho_theta(0,-1,-1,0))
