import torch
import torch.nn as nn
import torch.nn.functional as F


"""
class cls_loss(nn.Module):
	def __init__(self):
		super(cls_loss, self).__init__()

	def forward(self, gt_score, pred_score, ignore):
		pos_area = gt_score
		neg_area = (1-ignore) * (1-gt_score)
		pos_loss = -torch.log(pred_score + 1e-8) * pos_area
		neg_loss = -torch.log(1-pred_score + 1e-8) * neg_area

		total_loss = pos_loss.sum() + neg_loss.sum()
		return total_loss / (pos_area.sum() + neg_area.sum())
"""

class cls_loss(nn.Module):
	def __init__(self):
		super(cls_loss, self).__init__()
		self.ratio = 0

	def forward(self, gt_score, pred_score, ignore):
		pos_area = gt_score
		neg_area = (1-ignore) * (1-gt_score)
		pos_num = pos_area.sum()

		pos_loss = -torch.log(pred_score + 1e-8) * pos_area
		neg_loss = -torch.log(1-pred_score + 1e-8) * neg_area

		mean = neg_loss.sum().item() / neg_area.sum().item()
		mean = mean + self.ratio * (neg_loss.max().item() - mean)
		zero = torch.zeros_like(gt_score).to(gt_score.device)

		neg_area = torch.where(neg_loss > mean, neg_area, zero)
		neg_loss = torch.where(neg_loss > mean, neg_loss, zero)
		neg_num = neg_area.sum()
 
		ohem_loss = pos_loss.sum() + neg_loss.sum()
		return ohem_loss / (neg_num + pos_num)


class Loss_OHEM(nn.Module):
	def __init__(self):
		super(Loss_OHEM, self).__init__()
		self.cls_loss = cls_loss()

	def forward(self, text, ignore, rho, theta, pred_cls, pred_rho, pred_theta):
		cls_loss = self.cls_loss(text, pred_cls, ignore)
		"""
		rho_map = torch.abs(rho - pred_rho)
		"""

		mask = text.repeat(1, 4, 1, 1)

		rho_map = -torch.log(torch.min(pred_rho, rho) / (torch.max(pred_rho, rho) + 1e-8) + 1e-8)
		rho_loss = torch.sum(rho_map * mask) / (torch.sum(mask) + 1e-8)

		theta_loss = torch.sum(torch.sin(torch.abs(pred_theta - theta) * mask / 2)) / (torch.sum(mask) + 1e-8)

		print('cls {:.5f}, rho {:.5f}, theta {:.5f}'.format(cls_loss, rho_loss, theta_loss))
		loss = 1 * cls_loss + 1 * rho_loss + 1 * theta_loss
		return loss


