import argparse
import os
import warnings
warnings.filterwarnings("ignore")
import threading

#custom
from dataset import  GrayDataset
from utils import Plotter, GradCam
from runner import TSRunner

test_img_dir = r"D:\tsungyu\chromosome_data\chang\changValSum\images"
test_truth_dir = r"D:\tsungyu\chromosome_data\chang\changValSum\masks"
# test_img_dir = r"D:\tsungyu\chromosome_data\chang\chang_val_4\images"
# test_truth_dir = r"D:\tsungyu\chromosome_data\chang\chang_val_4\masks"


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: bn_unet, in_unet')
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')
    parser.add_argument('--fix_encoder', action="store_true",default=True, help='fix encoder')
    parser.add_argument('--pad_mode', action="store_true",default=True, help='unet used crop or pad at skip connection')
    parser.add_argument('--instanceloss', action="store_true",default=False, help='In this project, is always False')
    parser.add_argument('--in_channel','-i',type=int, default=1,help="channels of input images")
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--device', type=str,default='cpu',help='use cpu or gpu')
    parser.add_argument('--weight', '-w', default=r'log\IN_ASL_m0999_entropyweight_hardlabel\student_50.pth', metavar='FILE',help='Specify the file in which the model is stored')
    parser.add_argument('--imgdir', type=str,default=r'', help='the path of directory which imgs saved for predicting')
    parser.add_argument('--imgpath', '-img',type=str,default=r'', help='the path of img')
    parser.add_argument('--eval', action="store_true",default=True, help='calculate miou and dice score')
    
    return parser.parse_args()

def main():
    args = get_args()
    testset = GrayDataset(img_dir = test_img_dir, mask_dir = test_truth_dir)
    runner = TSRunner(args=args)
    runner.pretrain_model(weightPath=args.weight)
    modelName = os.path.basename(args.weight).split("_")[0]
    print(modelName)

    if args.imgdir:
        imgs = os.listdir(args.imgdir)
        imgs = [os.path.join(args.imgdir,i)for i in imgs]
        for i in imgs:
            runner.inference(i, modelName=modelName, entropymap=True)
    # predict one images
    elif args.imgpath:
        runner.inference(args.imgpath, modelName=modelName, entropymap=True)

    # evaluate images
    if args.eval:
        runner.evaluate(testdataset=testset, modelName=modelName)


if __name__ == '__main__':
    main()
    