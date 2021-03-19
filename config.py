from easydict import EasyDict as edict

cfg = edict()

cfg.pretrain = edict()
cfg.pretrain.mode = 'pretrain'
cfg.pretrain.img_path = '../data/SynthText'
cfg.pretrain.gt_path = '../data/SynthText'
cfg.pretrain.imagenet_pretrain = True
cfg.pretrain.pretrained_pth = None 
cfg.pretrain.scale = 0.25
cfg.pretrain.min_side = 640
cfg.pretrain.max_side = 1920
cfg.pretrain.crop_length = 640
cfg.pretrain.mean = [0.485, 0.456, 0.406]
cfg.pretrain.std = [0.229, 0.224, 0.225]
cfg.pretrain.backbone = '50' # '34'
cfg.pretrain.batch_size = 64
cfg.pretrain.num_workers = 8
cfg.pretrain.drop_last = True
cfg.pretrain.high_lr = 0.001
cfg.pretrain.low_lr = 0.0001
cfg.pretrain.weight_decay = 0
cfg.pretrain.epoch_iter = 5
cfg.pretrain.save_interval = 1
cfg.pretrain.pths_path = './pths/pretrain'

cfg.train = edict()
cfg.train.mode = 'finetune'
cfg.train.img_path = ['../data/ICDAR2017/train_img', '../data/ICDAR2017/valid_img']
cfg.train.gt_path = ['../data/ICDAR2017/train_gt', '../data/ICDAR2017/valid_gt']
cfg.train.imagenet_pretrain = False
cfg.train.pretrained_pth = './pths/pretrain/model_epoch_5.pth'
cfg.train.scale = 0.25
cfg.train.min_side = 640
cfg.train.max_side = 2560
cfg.train.crop_length = 640
cfg.train.mean = [0.485, 0.456, 0.406]
cfg.train.std = [0.229, 0.224, 0.225]
cfg.train.backbone = '50' # '34'
cfg.train.batch_size = 64
cfg.train.drop_last = True
cfg.train.num_workers = 8
cfg.train.high_lr = 0.001
cfg.train.low_lr = 0.0001
cfg.train.weight_decay = 0
cfg.train.epoch_iter = 150
cfg.train.save_interval = 1
cfg.train.pths_path = './pths/train'

cfg.test = edict()
cfg.test.short_side = 1280
cfg.test.rm_small_cc_area = 0.00
cfg.test.model_pth = './pths/train/model_epoch_150.pth'
cfg.test.dataset_test = '../data/ICDAR2017/test_img'
cfg.test.cls_thresh = 0.4
cfg.test.nms_thresh = 0.1
cfg.test.scale = 0.25
