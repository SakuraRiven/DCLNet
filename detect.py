import os
import cv2
import math
import numpy as np
import torch
import itertools
import pdb
import time
from torchvision import transforms
from PIL import Image, ImageDraw
from model import ResNetUNet
from config import cfg
import lanms


def resize_img(img, short_side):
    if short_side is None:
        return img

    w, h = img.size
    if w < h:
        resize_w = short_side
        ratio = short_side / w
        resize_h = h * ratio
    else:
        resize_h = short_side
        ratio = short_side / h
        resize_w = w * ratio
    if max(resize_w, resize_h) > 5000:
        return img
    img = img.resize((int(resize_w), int(resize_h)), Image.BILINEAR)
    return img


def load_pil(img):
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(cfg.train.mean, cfg.train.std)])
    return t(img).unsqueeze(0)


def get_score(img, model, device):
    with torch.no_grad():
        pred_cls, pred_rho, pred_theta = model(load_pil(img).to(device))
    pred_cls, pred_rho, pred_theta = list(map(lambda x: x[0].cpu().numpy(), [pred_cls, pred_rho, pred_theta]))
    return pred_cls, pred_rho, pred_theta


def line_cross_point(A1, B1, C1, A2, B2, C2):
    """
    F0(x) = a0*x + b0*y + c0 = 0, F1(x) = a1*x + b1*y + c1 = 0
    x = (b0*c1 – b1*c0)/D
    y = (a1*c0 – a0*c1)/D
    D = a0*b1 – a1*b0， (D为0时，表示两直线重合)
    """
    D = A1*B2 - A2*B1
    x = (B1*C2 - B2*C1) / (D + 1e-8)
    y = (A2*C1 - A1*C2) / (D + 1e-8)
    return x, y

def restore_bboxes(cls, rho, theta, cls_thresh=None, nms=True):
    # AX+BY+C=0  <---> ρ=x*cosθ+y*sinθ
    height = cls.shape[1]
    if cls_thresh is None:
        cls_thresh = cfg.test.cls_thresh

    region = (cls[0,:,:] > cls_thresh)
    ret, markers = cv2.connectedComponents(np.uint8(region))
    for i in range(1, ret):
        y, x = np.where(markers == i)
        if len(x) > (cfg.test.rm_small_cc_area*cfg.test.short_side)**2:
            continue
        for j in range(len(x)):
            region[y[j], x[j]] = False

    r, c = np.where(region)
    A, B, C = np.cos(theta), np.sin(theta), -rho # the 'C' is actually "C + A*dx + B*dy"
    A0, B0, C0 = A[0,r,c], B[0,r,c], C[0,r,c]; C0 -= A0*c + B0*(height-r) # recover C in global coordinate
    A1, B1, C1 = A[1,r,c], B[1,r,c], C[1,r,c]; C1 -= A1*c + B1*(height-r) # recover C in global coordinate
    A2, B2, C2 = A[2,r,c], B[2,r,c], C[2,r,c]; C2 -= A2*c + B2*(height-r) # recover C in global coordinate
    A3, B3, C3 = A[3,r,c], B[3,r,c], C[3,r,c]; C3 -= A3*c + B3*(height-r) # recover C in global coordinate

    # lines -> bboxes
    x1, y1 = line_cross_point(A0, B0, C0, A1, B1, C1)
    x2, y2 = line_cross_point(A1, B1, C1, A2, B2, C2)
    x3, y3 = line_cross_point(A2, B2, C2, A3, B3, C3)
    x0, y0 = line_cross_point(A3, B3, C3, A0, B0, C0)
    bboxes = np.vstack((x0, height-y0, x1, height-y1, x2, height-y2, x3, height-y3, cls[0,r,c])).T
    if nms:
        bboxes = lanms.merge_quadrangle_n9(bboxes.astype('float32'), cfg.test.nms_thresh)
    return bboxes


def detect_single_image(img, model, device):
    h, w = img.height, img.width
    img = resize_img(img, cfg.test.short_side)
    cls, rho, theta = get_score(img, model, device)
    bboxes = restore_bboxes(cls, rho, theta)
    bboxes[:,:8] = bboxes[:,:8] * 4 * (w / img.width + h / img.height) / 2 # restore scale and resize
    return bboxes


if __name__ == '__main__':
    img_files = [os.path.join(cfg.test.dataset_test, img_file) for img_file in sorted(os.listdir(cfg.test.dataset_test))]
    img_path = np.random.choice(img_files)
    print(img_path)

    model_path = cfg.test.model_pth
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNetUNet(backbone='50')
    model.load_state_dict(torch.load(model_path))
    model.to(device)

    model.eval()
    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    bboxes = detect_single_image(img, model, device)
    print(bboxes.shape)

    for bbox in bboxes:
        pts = bbox[:8].astype(np.int32).reshape((-1,1,2))
        cv2.polylines(img_array, [pts], True, (0,215,255), 3)
    Image.fromarray(img_array).save('res.png')



