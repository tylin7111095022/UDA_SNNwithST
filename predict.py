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

#online
from models.other_network import R2U_Net,NestedUNet
#custom
from models.unet_model import UNet
from dataset import RGBDataset, GrayDataset
from metric import iou,compute_mIoU
from utils import Plotter

test_img_dir = r"data\changgung_val\images"
test_truth_dir = r"data\changgung_val\masks"

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--weight', '-w', default=r'weights\in1_out2_inputpixel1\bestmodel.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--imgpath', '-img',type=str,default=r'A225314_01-01_040822144724_11_2.png', help='the path of img')
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
    net = UNet(n_channels =1,n_classes = 2)

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
