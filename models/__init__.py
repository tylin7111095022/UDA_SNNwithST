# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .IN_Unet import InstanceNormalization_UNet
import torch

def get_models(model_name:str, is_cls:bool, args):
    """option: in_unet, bn_unet"""
    if model_name == "in_unet":
        model = InstanceNormalization_UNet(n_channels=args.in_channel,n_classes=args.classes,is_normalize=args.is_normalize,is_cls=is_cls,pad_mode=args.pad_mode,instance_branch=args.instanceloss)
    else:
        raise NotImplementedError(f"{model_name} has not implemented")

    return model

if __name__ == '__main__':

    student = InstanceNormalization_UNet(n_channels=1, n_classes=2, is_cls=True)
    teacher = InstanceNormalization_UNet(n_channels=1, n_classes=2,is_proj=False, is_cls=True)
    load_path = r'weight\in\data10000\bestmodel.pth'
    student_name = { param[0]:param[1].data.detach() for param in student.named_parameters()}
    # teacher_name = { param[0]:param[1].data for param in teacher.named_parameters()}

    pretrained_model_param_dict = torch.load(load_path)
    student_param_dict = student.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict_s = {k: v for k, v in pretrained_model_param_dict.items() if k in student_param_dict}
    # 2. overwrite entries in the existing state dict
    student_param_dict.update(pretrained_dict_s)
    # 3. load the new state dict
    student.load_state_dict(student_param_dict)
    
    count = 0
    m = 0.99
    for param_t in teacher.named_parameters():
        print("-------------------------------")
        print(param_t[1])
        if param_t[0] in student_name.keys():
            count += 1
            param_t[1].data = param_t[1].data.mul_(m).add_((1-m)*student_name[param_t[0]])
        print("------------after--------------")
        print(param_t[1])
        

    print(len(student_name))
    # print(len(teacher_name))
    print(count)


        
