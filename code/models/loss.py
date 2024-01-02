import torch
from torch import nn


class Plane_CriterionCoordinate(nn.Module):
    def __init__(self):
        super(Plane_CriterionCoordinate, self).__init__()

    def forward(self, pred_point, gt_point, mask):
        diff_coord_map = torch.sum(torch.abs(pred_point - gt_point), axis=-1, keepdims=True)
        loss_map = mask * diff_coord_map
        valid_coord = torch.sum(mask)
        loss_map = (torch.sum(loss_map) + 1e-9) / (valid_coord + 1e-9)
        return loss_map


class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()

    def forward(self, pred_point, gt_point, mask):
        pred_pairwise_distance = torch.sum(torch.abs(pred_point[:, None, :] - pred_point[None, :, :]), dim=-1)
        gt_pairwise_distance = torch.sum(torch.abs(gt_point[:, None, :] - gt_point[None, :, :]), dim=-1)
        plane_pair_mask = mask @ mask.T
        distance_pair_mask = torch.abs(gt_pairwise_distance - pred_pairwise_distance) < 3
        pair_mask = plane_pair_mask*distance_pair_mask
        loss = torch.sum(pair_mask*torch.abs(pred_pairwise_distance - gt_pairwise_distance))
        valid_coord  = torch.sum(pair_mask)

        return loss, valid_coord

class CriterionCoordinate(nn.Module):
    def __init__(self):
        super(CriterionCoordinate, self).__init__()
        self.edge_loss = EdgeLoss()
        self.node_loss = Plane_CriterionCoordinate()

    def forward(self, pred_point, gt_point, mask, index):
        edge_loss = 0.0
        valid_coord       = 0
        for i in range(len(index)-1):
            batch_pred_point = pred_point[index[i]:index[i+1], :].view(-1, 3)
            batch_gt_point   = gt_point[index[i]:index[i+1], :].view(-1, 3)
            batch_mask       = mask[index[i]:index[i+1],:].view(-1, 1)
            batch_edge_loss, batch_valid_coord = self.edge_loss(batch_pred_point, batch_gt_point, batch_mask)
            edge_loss        += batch_edge_loss
            valid_coord      += batch_valid_coord

        node_loss = self.node_loss(pred_point, gt_point, mask)
        if valid_coord >= 1:
            edge_loss = edge_loss / valid_coord
            return node_loss + edge_loss
        else:
            return node_loss