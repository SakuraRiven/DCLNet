import os
import time
import numpy as np
import torch
import cv2
import re
import sys
from PIL import Image
from model import ResNetUNet
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torch.nn import DataParallel
from torch.optim import lr_scheduler
from config import cfg
from dataset import SynthTextDataset, FinetuneDataset
from loss import Loss_OHEM


def train(cfg):
	model = ResNetUNet(pretrain=cfg.imagenet_pretrain, backbone=cfg.backbone)
	if cfg.pretrained_pth:
		model.load_state_dict(torch.load(cfg.pretrained_pth, map_location='cpu'))
	if torch.cuda.device_count() > 1:
		model = DataParallel(model)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	trainset = SynthTextDataset(cfg) if cfg.mode == 'pretrain' else FinetuneDataset(cfg)
	train_loader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True, \
                                   num_workers=cfg.num_workers, pin_memory=False, drop_last=cfg.drop_last)

	file_num = len(trainset)
	batch_num = int(file_num/cfg.batch_size)
	criterion = Loss_OHEM()

	optimizer = optim.Adam(model.parameters(), lr=cfg.high_lr, weight_decay=cfg.weight_decay)
	scheduler = lr_scheduler.CosineAnnealingLR(optimizer, cfg.epoch_iter*batch_num, cfg.low_lr)

	model.train()
	for epoch in range(cfg.epoch_iter):	
		epoch_loss = 0
		epoch_time = time.time()
		for i, (img, text, ignore, rho, theta) in enumerate(train_loader):
			img, text, ignore, rho, theta = list(map(lambda x: x.to(device), [img, text, ignore, rho, theta]))
			pred_cls, pred_rho, pred_theta = model(img)
			loss = criterion(text, ignore, rho, theta, pred_cls, pred_rho, pred_theta)

			epoch_loss += loss.item()
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			scheduler.step()

			print('Epoch is [{}/{}], mini-batch is [{}/{}], batch_loss is {:.8f}'.format(\
               	epoch+1, cfg.epoch_iter, i+1, batch_num, loss.item()))
			sys.stdout.flush()

		if (epoch+1) % cfg.save_interval == 0:
			torch.save(model.module.state_dict(), os.path.join(cfg.pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
			print(time.asctime(time.localtime(time.time())))

		print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss/batch_num, time.time()-epoch_time))
		print(time.asctime(time.localtime(time.time())))
		print('='*50)


if __name__ == '__main__':
	seed = 1
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

	assert(len(sys.argv) == 2)
	if sys.argv[1] == 'pretrain':
		train(cfg.pretrain)
	elif sys.argv[1] == 'finetune':
		train(cfg.train)
	else:
		sys.exit(-1)

