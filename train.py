import argparse
import logging
import os
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")

#custom module
from models import get_models
from models.losses import Distribution_loss
from dataset import STDataset
from utils import adjust_lr, cosine_decay_with_warmup, set_grad, generate_class_mask, mix, prob2entropy

dir_img = r'D:\tsungyu\chromosome_data\cyclegan_data\fake_zong' #訓練集的圖片所在路徑 輸入到student network
dir_truth = r'D:\tsungyu\chromosome_data\cyclegan_data\real_chang' #訓練集的真實label所在路徑 輸入到teacher network
dir_checkpoint = r'log\discussion_after_test\IN_ASL_exchange_input_m09996' #儲存模型的權重檔所在路徑
load_path = r'D:\tsungyu\AdaIN_domain_adaptation\weights\in\data10000_100epoch\bestmodel.pth'

def get_args():
    parser = argparse.ArgumentParser(description = 'Train the UNet on images and target masks')
    parser.add_argument('--image_channel','-i',type=int, default=1,dest='in_channel',help="channels of input images")
    parser.add_argument('--total_epoch','-e',type=int,default=50,metavar='E',help='times of training model')
    parser.add_argument('--warmup_epoch',type=int,default=0,help='warm up the student model')
    parser.add_argument('--batch','-b',type=int,dest='batch_size',default=1, help='Batch size')
    parser.add_argument('--classes','-c',type=int,default=2,help='Number of classes')
    parser.add_argument('--init_lr','-r',type = float, default=0.02,help='initial learning rate of model')
    parser.add_argument('--device', type=str,default='cuda:0',help='training on cpu or gpu')
    parser.add_argument("--momentum", "-m", type=float, default=0.9996, help="momentum parameter for updating teacher model.")
    parser.add_argument('--loss', type=str,default='asl',help='loss metric, options: [kl_divergence, cross_entropy, dice_loss, mae, asl]')
    parser.add_argument('--is_loss_weight', action="store_true",default=False,help='weight at every pixel added to calculate loss')
    parser.add_argument('--model', type=str,default='in_unet',help='models, option: simclr, in_unet')
    parser.add_argument('--pad_mode', action="store_true",default=True, help='unet used crop or pad at skip connection') # pretrained model , pad mode == True
    parser.add_argument('--normalize', action="store_true",dest="is_normalize",default=True, help='model normalize layer exist or not')
    parser.add_argument('--fix_encoder', action="store_true",default=True, help='fix encoder')
    parser.add_argument('--instanceloss', action="store_true",default=False, help='In this project, is always False')
    parser.add_argument('--classmix', action="store_true",default=False, help='calculate miou')

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
    student = get_models(model_name=args.model, is_cls=True,args=args)
    teacher = get_models(model_name=args.model, is_cls=True,args=args)

    pretrained_model_param_dict = torch.load(load_path)
    student_param_dict = student.state_dict()
    teacher_param_dict = teacher.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict_s = {k: v for k, v in pretrained_model_param_dict.items() if k in student_param_dict}
    pretrained_dict_t = {k: v for k, v in pretrained_model_param_dict.items() if k in teacher_param_dict}
    # 2. overwrite entries in the existing state dict
    student_param_dict.update(pretrained_dict_s)
    teacher_param_dict.update(pretrained_dict_t)
    # 3. load the new state dict
    student.load_state_dict(student_param_dict)
    teacher.load_state_dict(teacher_param_dict)
    
    logging.info(student)
    logging.info("="*20)
    logging.info(teacher)
    optimizer = torch.optim.Adam(student.parameters(),lr = args.init_lr,betas=(0.9,0.999))
    ##紀錄訓練的一些參數配置
    logging.info(f'''
    =======================================
    student and teacher model are both initialized by zong weights.
    update teacher model(EMA) at every \'epoch\' too. 
    
    dir_img: {dir_img}
    dir_truth: {dir_truth}
    dir_checkpoint: {dir_checkpoint}
    teacher_load_path : {load_path}
    args: 
    {args}
    =======================================
    ''')
    try:
        training(net=student,
                 teacher=teacher,
                optimizer = optimizer,
                dataset = trainingDataset,
                args=args,
                save_checkpoint= True,)
                
    except KeyboardInterrupt:
        torch.save(student.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        raise

    return

def training(net,
             teacher,
             optimizer,
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
    net.to(device)
    teacher.to(device)
    set_grad(model=teacher, is_requires_grad=False)
    loss_fn = Distribution_loss()
    loss_fn.set_metric(args.loss)
    #begin to train model
    epoch_losses = []
    for i in range(1, args.total_epoch+1):
        net.train()
        teacher.train()
        
        epoch_loss = 0
        # adjust the learning rate
        lr = cosine_decay_with_warmup(current_iter=i,total_iter=args.total_epoch,warmup_iter=args.warmup_epoch,base_lr=args.init_lr)
        adjust_lr(optimizer,lr)

        for imgs, imgs_src_style in tqdm(train_loader):

            imgs = imgs.to(device=device,dtype=torch.float32)
            # h,w = imgs.shape[2], imgs.shape[3]
            imgs_src_style = imgs_src_style.to(device=device,dtype=torch.float32)
            logit_s = net(imgs)
            logit_t = teacher(imgs_src_style)

            hard_label = torch.zeros_like(logit_t)
            index = torch.argmax(torch.softmax(logit_t.detach(),dim=1),dim=1,keepdim=True)
            hard_label.scatter_(1, index, 1)

            # print(hard_label.shape)
            # print(hard_label[:,0,1,1])
            # print(hard_label[:,1,1,1])

            # calcullate weight of every pixel im the image
            weight = None
            if args.is_loss_weight:
                entmap = prob2entropy(torch.softmax(logit_t.detach(),dim=1)) #上下限0~1
                entmap = torch.where(torch.isnan(entmap),torch.full_like(entmap,0),entmap) # NaN 補 0 # entropy高 權重越高
                # print(entmap.min(),entmap.max())
                weight = torch.ones_like(entmap) - entmap # entropy低 權重越高

            loss = loss_fn(hard_label, logit_s,weight)
            if args.classmix:
                with torch.no_grad():
                    teacher_predict = torch.argmax(torch.softmax(logit_t.detach(),dim=1),dim=1)
                    student_predict = torch.argmax(torch.softmax(logit_s.detach(),dim=1),dim=1)
                    mask = generate_class_mask(pred=teacher_predict,classes=torch.ones(1),device=device)
                    mix_img, mix_label = mix(mask,sourcedata=imgs,targetdata=imgs_src_style,sourcelabel=student_predict,targetlabel=teacher_predict)
                logit_mix = net(mix_img)
                loss_aux = CrossEntropyLoss()(logit_mix, mix_label.squeeze(1))
                loss += loss_aux
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # update teacher model and it also apply the situation which  both architectures of student model and teacher model are different
        with torch.no_grad():
            m = args.momentum  # momentum parameter
            student_name_parameters = { param[0]:param[1].data.detach() for param in net.named_parameters()}
            for param_t in teacher.named_parameters():
                if param_t[0] in student_name_parameters.keys():
                    param_t[1].data = param_t[1].data.mul_(m).add_((1-m)*student_name_parameters[param_t[0]])

        logging.info(f'Training loss: {epoch_loss:6.4f} at epoch {i}.')
        epoch_losses.append(epoch_loss)

        if (save_checkpoint) :
            torch.save(net.state_dict(), os.path.join(dir_checkpoint,f'student_{i}.pth'))
            torch.save(teacher.state_dict(), os.path.join(dir_checkpoint,f'teacher_{i}.pth'))
            logging.info(f'Model saved at epoch {i}.')
        
    min_loss_at = torch.argmin(torch.tensor(epoch_losses)).item() + 1 
    logging.info(f'min Training loss at epoch {min_loss_at}.')
            
    return

if __name__ == '__main__':
    main()