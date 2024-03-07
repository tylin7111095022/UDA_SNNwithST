import torch
import sys
import numpy as np
from sklearn.metrics import confusion_matrix
'''
dice_score = (2*precision*recall)/ (precision + recall) 
dice_score越接近1越好

https://chih-sheng-huang821.medium.com/%E5%BD%B1%E5%83%8F%E5%88%87%E5%89%B2%E4%BB%BB%E5%8B%99%E5%B8%B8%E7%94%A8%E7%9A%84%E6%8C%87%E6%A8%99-iou%E5%92%8Cdice-coefficient-3fcc1a89cd1c
'''

SMOOTH = sys.float_info.min

def iou(outputs: torch.Tensor, labels: torch.Tensor):
    """計算某一類別的iou值"""
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x H x W shape
    assert outputs.dim() == labels.dim(), "dim dismatch"
    assert outputs.dim() == 3, "dim must be 3 , (BHW)"
    inter = torch.sum((outputs & labels).float()).item()
    union = torch.sum((outputs | labels).float()).item()
    iou_score = (inter + SMOOTH) / (union + SMOOTH)
    
    return iou_score  

def compute_mIoU(pred, label):
    """計算整體類別的平均iou"""
    assert len(pred.shape) == len(label.shape), f"dim dismatch, pred shape {len(pred.shape)} is not equal label shape {len(label.shape)}."
    assert len(pred.shape) == 4, "dim must be 4 , (BCHW)"
    if pred.shape[1] == 1: #預測的map已經變為一張圖了
        cf_m = confusion_matrix(label.flatten(), pred.flatten())
        # print(f"confusion matrix \n {cf_m}")
        intersection = np.diag(cf_m)  # TP + FN
        union = np.sum(cf_m, axis=1) + np.sum(cf_m, axis=0) - intersection #模型預測全是某類的值 + 實際真的是該類的值 - 正確預測的值
        # print("inter", intersection)
        # print("union", union)
        IoU = intersection / union
        mIoU = np.nanmean(IoU) # Compute the arithmetic mean along the specified axis, ignoring NaNs.

    return mIoU

def dice_score(pred, label):
    assert len(pred.shape) == len(label.shape), f"dim dismatch, pred shape {len(pred.shape)} is not equal label shape {len(label.shape)}."
    # assert len(pred.shape) == 4, "dim must be 4 , (BCHW)"
    tp = torch.sum(pred*label)
    fp_and_fn = torch.sum(torch.logical_or(pred, label))
    dice = (2*tp) / (2*tp + fp_and_fn)
    return dice.item()

if __name__ == "__main__":
    pred = torch.tensor([[[0,0,1],
                         [1,1,0],
                         [1,0,1]]])

    label = torch.tensor([[[1,0,1],
                          [0,1,0],
                          [1,0,1]]])

    miou = compute_mIoU(pred.unsqueeze(0), label.unsqueeze(0))
    print(miou)
    iou = iou(pred,label)
    print(iou)
    dice = dice_score(pred, label)
    print("dice",dice)
