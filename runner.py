import torch
from models.losses import Distribution_loss
from models import get_models
from utils import set_grad, prob2entropy
import json
import os


class BaseRunner(object):
    def __init__(self, teacher, student, loss, optimizer, args):
        self.teacher = teacher
        self.student = student
        self.loss = loss
        self.optimizer = optimizer
        self._args = args
    
    def train(self):
        pass
    
    def inference(self):
        pass

    def update_param(self):
        pass

    def set_args(self, args:dict):
        self._args = args

    def get_cfg(self, filename:str = 'train_cfg.json'):
        print(self._args)
        with open(filename, "w") as outfile: 
            json.dump(self._args, outfile)


class TSRunner(BaseRunner):
    def __init__(self, args):
        self._args = args
        self.teacher = get_models(args.model, is_cls=True,args=args)
        self.student = get_models(args.model, is_cls=True,args=args)
        self.lossfn = Distribution_loss()
        self.optimizer = None
        self.loss_val = 0
        # initialize
        self.set_args()
    
    def inference(self):
        self.teacher.eval()
        self.student.eval()
        pass

    def update_param(self, dataForTeacher, dataForStudent, needWeight:bool=False):
        logits, labels, weights = self._forward(dataForTeacher, dataForStudent, needWeight)
        loss = self.lossfn(labels, logits, weights)
        self.loss_val = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # update student
        self._update_teacher_param(self._args.momentum) # update teacher

    def set_args(self, args= None):
        self._args = args if args is not None else self._args
        self.loss_val = 0
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr = self._args.init_lr,betas=(0.9,0.999))
        self.lossfn.set_metric(self._args.loss)
        self.pretrain_model(self._args.loadpath)
        set_grad(model=self.teacher, is_requires_grad=False)
        self.teacher.to(self._args.device)
        self.student.to(self._args.device)

    def pretrain_model(self, weightPath = None):
        if (weightPath is None):
             return
        pretrained_model_param_dict = torch.load(weightPath)
        student_param_dict = self.student.state_dict()
        teacher_param_dict = self.teacher.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict_s = {k: v for k, v in pretrained_model_param_dict.items() if k in student_param_dict}
        pretrained_dict_t = {k: v for k, v in pretrained_model_param_dict.items() if k in teacher_param_dict}
        # 2. overwrite entries in the existing state dict
        student_param_dict.update(pretrained_dict_s)
        teacher_param_dict.update(pretrained_dict_t)
        # 3. load the new state dict
        self.student.load_state_dict(student_param_dict)
        self.teacher.load_state_dict(teacher_param_dict)

    def save_weight(self, saveFolder, suffix):
        torch.save(self.student.state_dict(), os.path.join(saveFolder,f'student_{suffix}.pth'))
        torch.save(self.teacher.state_dict(), os.path.join(saveFolder,f'teacher_{suffix}.pth'))
    
    def _forward(self, dataForTeacher, dataForStudent, needWeight:bool=False):
        weight = None
        self.teacher.train()
        self.student.train()
        dataForTeacher = dataForTeacher.to(device=self._args.device,dtype=torch.float32)
        dataForStudent = dataForStudent.to(device=self._args.device,dtype=torch.float32)
        # Generate pseudoLabel
        logit_t = self.teacher(dataForTeacher)
        logit_t = logit_t.detach()
        hard_label = torch.zeros_like(logit_t)
        index = torch.argmax(torch.softmax(logit_t.detach(),dim=1),dim=1,keepdim=True)
        hard_label.scatter_(1, index, 1)

        if needWeight:
            entmap = prob2entropy(torch.softmax(logit_t.detach(),dim=1)) #上下限0~1
            entmap = torch.where(torch.isnan(entmap),torch.full_like(entmap,0),entmap) # NaN 補 0 # entropy高 權重越高
            # print(entmap.min(),entmap.max())
            weight = torch.ones_like(entmap) - entmap # entropy低 權重越高

        logit_s = self.student(dataForStudent)

        return logit_s, hard_label, weight
    
    def _update_teacher_param(self, momentum:float):
        # update teacher model and it also apply the situation which  both architectures of student model and teacher model are different
        with torch.no_grad():
            student_name_parameters = { param[0]:param[1].data.detach() for param in self.student.named_parameters()}
            for param_t in self.teacher.named_parameters():
                if param_t[0] in student_name_parameters.keys():
                    param_t[1].data = param_t[1].data.mul_(momentum).add_((1-momentum)*student_name_parameters[param_t[0]])
