import argparse
import os
import cv2
import numpy as np
import torch
import warnings
from PIL import Image
warnings.filterwarnings("ignore")
from tqdm import tqdm
import threading

#custom
from models import get_models
from dataset import RGBDataset, GrayDataset
from metric import iou,compute_mIoU
from utils import Plotter

test_img_dir = r"data\chang_val_1\images"
test_truth_dir = r"data\chang_val_1\masks"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: bn_unet, in_unet')
    parser.add_argument('--in_channel','-i',type=int, default=1,help="channels of input images")
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')

    parser.add_argument('--weight', '-w', default=r'log\train15_byol_in\student_50.pth', metavar='FILE',help='Specify the file in which the model is stored')
    parser.add_argument('--imgpath', '-img',type=str,default=r'', help='the path of img')
    parser.add_argument('--miou', action="store_true",default=True, help='calculate miou')
    
    return parser.parse_args()

def predict_mask(net,imgpath:str):
    plotter = Plotter()
    net = net.to(device="cpu")
    img = torch.from_numpy(cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)).unsqueeze(2).permute(2,0,1)
    img = img.unsqueeze(0)#加入批次軸
    img = img.to(dtype=torch.float32, device='cpu')
    mask_pred_prob = net(img)
    plotter.plot_entropy(torch.softmax(mask_pred_prob, dim=1),saved=True,is_heat=True)
    mask_pred = torch.argmax(torch.softmax(mask_pred_prob, dim=1),dim=1,keepdim=True).to(torch.int32)
    mask_pred = mask_pred.squeeze().numpy()
    mask_pred = mask_pred.astype(np.uint8)*255
    im = Image.fromarray(mask_pred)
    im.save(f"./predict_{os.path.basename(imgpath)}")

    return mask_pred

def evaluate_imgs(net,
                testdataset,):
    # net.eval() #miou 計算不用 eval mode 因為 running mean and running std 誤差可能在訓練過程紀錄的時候過大
    total_iou = 0
    count = 0
    miou_list = []
    for i,(img, truth) in enumerate(tqdm(testdataset)):
        img = img.unsqueeze(0)#加入批次軸
        img = img.to(dtype=torch.float32)
        truth = truth.unsqueeze(0).to(dtype=torch.int64)#加入批次軸
        #print('shape of truth: ',truth.shape)
        with torch.no_grad():
            mask_pred_prob = net(img)   
            mask_pred = torch.argmax(torch.softmax(mask_pred_prob,dim=1),dim=1,keepdim=True).to(torch.int64) # (1,1,h ,w)
            #print('shape of mask_pred: ',mask_pred.shape)
            # mask = mask_pred.squeeze(0).detach()#(1,h ,w)
            # mask *= 255 #把圖片像素轉回255
            #compute the mIOU
            miou = compute_mIoU(mask_pred.numpy(), truth.numpy())
            # print(miou)
            miou_list.append(miou)
            # print('Mean Intersection Over Union: {:6.4f}'.format(miou))

    # return total_iou / count #回傳miou
    return sum(miou_list) / len(miou_list)

if __name__ == '__main__':
    args = get_args()
    testset = GrayDataset(img_dir = test_img_dir, mask_dir = test_truth_dir)
    if os.path.basename(args.weight).split("_")[0] == "student":
        net = get_models(model_name=args.model,is_proj=True, is_cls=True,args=args)
    else:
        net = get_models(model_name=args.model,is_proj=False, is_cls=True,args=args)

    print(f'Loading model {args.weight}')
    net.load_state_dict(torch.load(args.weight, map_location="cpu",))
    print('Model loaded!')
    # predict one images
    if args.imgpath:
        predict_mask(net=net,imgpath=args.imgpath)

    # evaluate images
    if args.miou:
        miou = evaluate_imgs(net=net,testdataset=testset)
        print(f'miou = {miou:6.4f}')
