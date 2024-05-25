import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import json
import torch
import torch.nn as nn
from typing import Optional

class GradCam(object):
    def __init__(self, model:nn.Module, gradCamLayer:Optional[list]=None):
        self.model = model
        self.targetLayers = gradCamLayer
        self.activationAndGrads = ActivationandGradients(model,gradCamLayer)
        self.device = next(self.model.parameters()).device

    def forwardandBackward(self, x, class_ndx:int):
        self.model.zero_grad()
        logits = self.activationAndGrads(x)
        prob = torch.softmax(logits,dim=1)
        masks = prob.argmax(dim=1)
        totalscore = torch.sum(prob[:,class_ndx,:,:] * masks)
        totalscore.backward()
        return

    def mode(self,mode:str):
        if mode == "train":
            self.model.train()
        elif mode == "eval":
            self.model.eval()
        else:
            print("Error, mode should be train or eval !!!, mode is eval now.")
            self.model.eval()

    def to(self, device:str):
        self.model.to(device=device)
        self.device = next(self.model.parameters()).device

    def load_weights(self, weights):
        model_dict = self.model.state_dict()
        pretrained_w_dict = torch.load(weights)
        weights_dict = {k:v for k, v in pretrained_w_dict.items() if k in model_dict }
        self.model.load_state_dict(weights_dict)
    
    def get_cam_images(self,x, class_ndx:int):
        x = x.to(self.device,dtype=torch.float32)
        self.forwardandBackward(x,class_ndx)
        target_size = (x.shape[-2], x.shape[-1])
        cam_per_target_layer = []
        weights = []
        cams = []
        # get cam weights
        for grad in self.activationAndGrads.gradients:
            weight = torch.mean(grad, dim=(2,3))
            weights.append(weight)
        # get cam activations
        activations = self.activationAndGrads.activations
        # print(f"activations: {activations}")
        for w, a in zip(weights, activations):
            weighted_activations = w[:, :, None, None] * a
            cam = torch.sum(weighted_activations,dim=1)
            cam = torch.where(cam > 0., cam, 0.)  # relu, cam shape: [b,h,w]
            cams.append(cam)
        for cam in cams:
            scaled_cam = self._scale_cam_image(cam, target_size)
            cam_per_target_layer.append(scaled_cam[:,None,:,:]) # insert channel axis shape: [b,1,h,w]

        return cam_per_target_layer

    def _scale_cam_image(self, cams, target_size=None):
        result = []
        for img in cams: # img shape [h,w]
            img = img - torch.min(img)
            img = img / (1e-7 + torch.max(img)) # normalize
            img_a = img.numpy()
            if target_size is not None:
                img = cv2.resize(img_a, target_size)
            result.append(img)
        result = np.stack(result,axis=0)
        result = np.float32(result) # result shape [b,h,w]

        return result
    
    def plot_cams(self, imgpath, class_ndx:int):
        name = os.path.basename(imgpath).split(".")[0]
        img_a = cv2.imread(imgpath,cv2.IMREAD_GRAYSCALE)
        img_t = torch.from_numpy(img_a)[None, None,:,:] # 1,1,h,w
        cams_per_target = self.get_cam_images(img_t,class_ndx)
        for i, cam in enumerate(cams_per_target):
            cam = cam.squeeze() # remove one dimension
            heatmap = np.uint8(255 * cam)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            superimposed_img = heatmap * 0.4 + img_a[:,:,None]
            cv2.imwrite(f'{name}_class{class_ndx}_{i}_gradcam.jpg', superimposed_img)

    
# ref: https://github.com/jacobgil/pytorch-grad-cam/tree/master
class ActivationandGradients(object):
    def __init__(self, model, targetLayer:list):
        self.model = model
        self.targetLayer = targetLayer
        self.gradients = []
        self.activations = []
        self.handles = []

        for target in targetLayer:
            h_a = target.register_forward_hook(self.get_activations)
            h_g = target.register_forward_hook(self.get_gradients)
            self.handles.append(h_a)
            self.handles.append(h_g)

    def get_activations(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def get_gradients(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            # You can only register hooks on tensor requires grad.
            return
        # Gradients are computed in reverse order
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        # initial gradients and activations
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()


class Plotter(object):
    def __init__(self):
        pass

    def plot_entropy(self,prob, saved:bool=False, is_heat:bool=True):
        ent_map = self._prob2entropy(prob)
        ent_map = torch.where(torch.isnan(ent_map),torch.full_like(ent_map,0),ent_map) # NaN 補 0
        maps = (ent_map - torch.min(ent_map)) / (torch.max(ent_map) - torch.min(ent_map))*255
        for m in range(maps.size(0)):
            map = maps[m].to(torch.uint8).permute(1,2,0).numpy()
            #黑白圖變熱圖
            if is_heat:
                map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
            if saved:
                cv2.imwrite(f"entropy_map{m}.png",map)
            else:
                cv2.imshow(f"entropy map{m}", map)
                cv2.waitKey(0)
                cv2.destroyAllWindows() 
        
    def _prob2entropy(self,prob):
        if isinstance(prob,np.ndarray):
            prob = torch.from_numpy(prob)
        if len(prob.shape) == 3:
            prob = prob.unsqueeze(0) # batch axis
        entropy = torch.mul(prob, -torch.log2(prob)) # 元素相乘
        ent_map = torch.sum(entropy,dim=1,keepdim=True)
        return ent_map

def calc_label_dist(root_dir:str):
    """計算每一類別(染色體)的mask中每個label(0,1)的分佈狀態
       root_dir: mask image 的所在路徑
    """
    masks = os.listdir(root_dir)
    record = {}
    for mask in masks:
        m_path = os.path.join(root_dir,mask)
        img = cv2.imread(m_path,cv2.IMREAD_UNCHANGED)
        ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
        # print(np.unique(img))
        tot_pixel = img.shape[0]*img.shape[1]
        print(f"tot_pixel: {tot_pixel}")
        background = np.sum((img[:,:] == 0))
        print(f"background: {background}")
        label = np.sum((img[:,:] == 255))
        print(f"label: {label}")
        assert tot_pixel == background + label, "pixel sum dismatch."
        label /= tot_pixel
        background /= tot_pixel
        record[m_path] = (background, label)
    with open(f"{os.path.basename(root_dir)}.json", "w") as f:
        json.dump(record, f,indent=4)
    return


def plot_img_and_mask(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[i, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def adjust_lr(optimizer, lr):
    for param in optimizer.param_groups:
        param["lr"] = lr

def cosine_decay_with_warmup(current_iter:int, total_iter:int, warmup_iter:int, base_lr:float):
    assert current_iter <= total_iter
    assert warmup_iter < total_iter

    if current_iter > warmup_iter:
        lr = 0.5 * base_lr * (1 + (np.cos(np.pi*(current_iter-warmup_iter)/(total_iter-warmup_iter))))
    else:
        slope = float(base_lr / warmup_iter)
        lr = slope * current_iter
    return lr

def restore_param(model, model_state, prob):
    for nm, module in model.named_modules():
        for name_p, param in module.named_parameters():
            if name_p in ['weight', 'bias']:
                mask = (torch.rand(param.shape)<prob).float().cuda() 
                with torch.no_grad():
                    param.data = model_state[f"{nm}.{name_p}"] * mask + param * (1.-mask)
    return model

def set_grad(model, is_requires_grad:bool):
    for p in model.parameters():
        p.requires_grad = is_requires_grad
    
    return model

def generate_class_mask(pred, classes, device): # in this case, classes always equal to 1
    # print("pred shape: ", pred.shape)
    # print("classes shape: ", classes.shape)
    pred, classes = torch.broadcast_tensors(pred.unsqueeze(0).to(device), classes.unsqueeze(1).unsqueeze(2).to(device)) # pred 鬆開channel軸， classes鬆開h,w軸
    N = torch.sum(pred.eq(classes),dim=0,keepdim=True)
    # print("mask shape: ", N.shape)
    return N # shape should be (1,h,w)

def mix(mask, sourcedata = None,targetdata = None, sourcelabel = None, targetlabel = None):
    # Mix
    mix_data = None
    mix_label = None
    if not (sourcedata is None or targetdata is None):
        mix_data = (mask * sourcedata + (1 - mask) * targetdata)

    if not (sourcelabel is None or targetlabel is None):
        mix_label = (mask * sourcelabel + (1 - mask) * targetlabel)
    # print("mix_data shape: ", mix_data.shape)
    # print("mix_label shape: ", mix_label.shape)
    return mix_data, mix_label

def prob2entropy(prob):
    if isinstance(prob,np.ndarray):
        prob = torch.from_numpy(prob)
    if len(prob.shape) == 3:
        prob = prob.unsqueeze(0) # batch axis
    entropy = torch.mul(prob, -torch.log2(prob)) # 元素相乘
    ent_map = torch.sum(entropy,dim=1,keepdim=True)
    return ent_map

if __name__ == "__main__":
    # main()
    lrs = []
    for i in range(1, 51):
        lr = cosine_decay_with_warmup(i, 50,0, 1e-1)
        lrs.append(lr)

    plt.plot(np.arange(1,51), lrs)
    plt.show()