
from __future__ import print_function
import copy
import  csv
import numpy as np
import os
import numpy
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.init as nninit
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
from vit_d import VisionTransformer

import time
from PIL import Image
import torchattacks2

use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")

ROOT = '..'

class AdvCIFAR10Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, root, transform):
        'Initialization'
        self.root = root
        self.transform = transform
        self.filelist = [file for file in os.listdir(root) if file.endswith('.png')]

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.filelist)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        filename = self.filelist[index]
        label = int(filename.split(".png")[0].split("_")[1])

        # Load data and get label
        image = Image.open(self.root + "/" + filename)
        imaget = self.transform(image)

        return imaget, label

adv_dataset = AdvCIFAR10Dataset(root='../adv_images/PGD_defense', transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=10, shuffle=False, num_workers=1)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR10(root=ROOT+"/data", train=False, transform=transforms.Compose([
                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]), download=True),
                        batch_size=10,
                        shuffle=False,
                        num_workers=1
                        )

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.model = VisionTransformer(
        img_size=32, patch_size=2, in_chans=3, num_classes=10, embed_dim=80, depth=20,
                 num_heads=20, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm)
    def forward(self,x):
        return self.model(x)

def linf_distance(images, adv_images):
    diff = abs(images-adv_images)
    maxlist = []
    for i in diff:
        maxlist.append(torch.max(i))
    return 255*((sum(maxlist)/len(maxlist)).item())

def l2_distance(corrects, images, adv_images, device="cuda"):
    delta = (adv_images - images.to(device)).view(len(images), -1)
    l2 = torch.norm(delta[~corrects], p=2, dim=1).mean()
    return l2.item()

def test(model,test_loader):
    model.eval()
    
    # IPGD2=final attack in the paper
    # IPGD=old unsuccessful attack
    # PGD=standard PGD algorithm specifically modified for normalization
    
    # Modified torchattack library to incorporate normalization requirements
    atk = torchattacks2.PGD(model, mean=0.5, std=0.5, eps=3/255., alpha=1/255., steps=12)
    #atk = torchattacks2.IPGD2(model, mean=0.5, std=0.5, eps=3/255., alpha=1/255., steps=12, alpha2=1/255, steps2=12)
    #atk = torchattacks2.IPGD(model, mean=0.5, std=0.5, eps=10/255., alpha=2/255., steps=20)
    
    print("-"*100)
    print(atk)
    
    correct = 0
    total = 0
    avg_act = 0
    counter = 0
    
    start = time.time()
    
    # adv_loader to test 100 CIFAR10 adversarial images
    # test_loader to test 100 CIFAR10 images
    for images, labels in test_loader:
        labels = labels.to(device)
        images = images.to(device)
        adv_images = images
        
        # Uncomment to attack loader images
        adv_images = atk(images, labels)
        
        # Saves adversarial examples
        normalized_imgs = images * 0.5 + 0.5
        normalized_aimgs = adv_images * 0.5 + 0.5
        
        # Uncomment to save images
        #for i in range(len(normalized_aimgs)):
        #    torchvision.utils.save_image(normalized_aimgs[i], ROOT+"/adv_images/PGD_defense/" + str(counter) + "_" + str(labels[i].item()) + ".png", normalize=False)
        #    counter = counter + 1
        
        total += len(labels)
        
        data16x16 = torch.nn.functional.interpolate(adv_images, size=(16, 16),mode='bilinear', align_corners=False)
        with torch.no_grad():
            out = torch.nn.Softmax(dim=1).cuda()(model(adv_images))
            out16x16 = torch.nn.Softmax(dim=1).cuda()(model(data16x16))
                    
        act,pred = out.max(1, keepdim=True)
        _,pred16x16 = out16x16.max(1, keepdim=True)
        corrects = (pred16x16.view_as(labels)==labels)
        correct += (pred16x16==labels.view_as(pred16x16))[pred16x16==pred].sum().cpu()
        avg_act += act.sum().data

        l2 = l2_distance(corrects, normalized_imgs, normalized_aimgs, device=device)
        linf = linf_distance(normalized_imgs, normalized_aimgs)
        print("Images: " + str(total) + " | Robust Acc: " + str(100* float(correct)/float(total)) + "% | L2 Distance: " + str(round(l2, 3)) + " | Linf Distance: " + str(round(linf)))
        
        #if total >= 100:
        #    break

    print('Total elapsed time (sec) : %.2f' % (time.time() - start))
    print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    
    return 100. * float(correct) / len(test_loader.dataset),100. * float(avg_act) / len(test_loader.dataset)


if __name__=="__main__":
        model = NN()
        model.cuda()

        if os.path.isfile("mdl.pth"):
            chk = torch.load("mdl.pth")
            model.load_state_dict(chk["model"]);
            del chk
        torch.cuda.empty_cache();
        acc,_ = test(model,test_loader)
        print('Test accuracy: ',acc)

