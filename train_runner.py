import argparse
import logging
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

#custom module
from runner import TSRunner
from dataset import STDataset
from utils import adjust_lr, cosine_decay_with_warmup

dir_img = r'D:\tsungyu\chromosome_data\cyclegan_data\fake_zong' #訓練集的圖片所在路徑 輸入到student network
dir_truth = r'D:\tsungyu\chromosome_data\cyclegan_data\real_chang' #訓練集的真實label所在路徑 輸入到teacher network
dir_checkpoint = r'log\discussion_after_test\exchange_testRunner' #儲存模型的權重檔所在路徑

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--total_epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--warmup_epoch',type=int,default=0,help='warm up the student model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--init_lr','-r',type = float, default=0.02,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    parser.add_argument("--momentum", "-m", type=float, default=0.99, help="momentum parameter for updating teacher model.")
    parser.add_argument('--loss', type=str,default='asl',help='loss metric, options: [kl_divergence, cross_entropy, dice_loss, mae, asl]')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: simclr, in_unet')
    parser.add_argument('--loadpath', type=str,default=r'D:\tsungyu\AdaIN_domain_adaptation\weights\in\data10000_100epoch\bestmodel.pth',help='pretrain model path')
    parser.add_argument('--is_loss_weight', action="store_true",default=True,help='weight at every pixel added to calculate loss')
    parser.add_argument('--pad_mode', action="store_true",default=True, help='unet used crop or pad at skip connection') # pretrained model , pad mode == True
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')
    parser.add_argument('--fix_encoder', action="store_true",default=True, help='fix encoder')
    parser.add_argument('--instanceloss', action="store_true",default=False, help='In this project, is always False')

    return parser.parse_args()

def main():
    args = get_args()
    trainingDataset = STDataset(student_dir = dir_img, teacher_dir= dir_truth)

    os.makedirs(dir_checkpoint,exist_ok=False)

    #設置 log
    # ref: https://shengyu7697.github.io/python-logging/
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(os.path.join(dir_checkpoint,"log.txt"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)
    ###################################################
    runner = TSRunner(args=args)
    runner.set_args()
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    student and teacher model are both initialized by zong weights.
    update teacher model(EMA) at every \'epoch\' too. 
    
    dir_img: {dir_img}
    dir_truth: {dir_truth}
    dir_checkpoint: {dir_checkpoint}
    
    args: 
    {args}
    =======================================
    ''')
    
    training(runner = runner,
            dataset = trainingDataset,
            args=args,
            save_checkpoint= True,)

    return

def training(runner,
             dataset,
             args,
             save_checkpoint: bool = True):

    arg_loader = dict(batch_size = args.batch_size, num_workers = 4)
    train_loader = DataLoader(dataset,shuffle = True, **arg_loader)
    device = torch.device( args.device if torch.cuda.is_available() else 'cpu')
    #Initial logging
    logging.info(f'''Starting training:
        model:           {args.model}
        Epochs:          {args.total_epoch}
        warm up epoch:   {args.warmup_epoch}
        Batch size:      {args.batch_size}
        Loss metirc      {args.loss}
        Training size:   {len(dataset)}
        checkpoints:     {save_checkpoint}
        Device:          {device.type}
    ''')

    #begin to train model
    epoch_losses = []
    for i in range(1, args.total_epoch+1):
        epoch_loss = 0
        # adjust the learning rate
        lr = cosine_decay_with_warmup(current_iter=i,total_iter=args.total_epoch,warmup_iter=args.warmup_epoch,base_lr=args.init_lr)
        adjust_lr(runner.optimizer,lr)
        for imgs, imgs_src_style in tqdm(train_loader):
            runner.update_param(imgs_src_style, imgs, args.is_loss_weight)
            epoch_loss += runner.loss_val

        logging.info(f'Training loss: {epoch_loss:6.4f} at epoch {i}.')
        epoch_losses.append(epoch_loss)

        if (save_checkpoint) :
            runner.save_weight(dir_checkpoint, i)
            logging.info(f'Model saved at epoch {i}.')
        
    min_loss_at = torch.argmin(torch.tensor(epoch_losses)).item() + 1 
    logging.info(f'min Training loss at epoch {min_loss_at}.')
            
    return

if __name__ == '__main__':
    main()