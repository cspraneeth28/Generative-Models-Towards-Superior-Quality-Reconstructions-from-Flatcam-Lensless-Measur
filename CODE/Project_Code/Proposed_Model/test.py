import torch
import scipy.io as sio
import numpy as np
import os
from skimage.color import rgb2gray
import skimage.io
import random
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as rsz
import torch.optim as optim
import os
from torch_vgg import Vgg16
from models import FlatNet
from models import Discriminator
from fns_all import*
from dataloader import*
import argparse
from torch.utils import data
import torchvision.transforms as transforms
import skimage.transform
import copy
import sys
import pprint
from datetime import datetime
from pytz import timezone
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

parser = argparse.ArgumentParser()
#model and data locs
parser.add_argument('--train_meas_filenames', default='./casia_file_names_train.txt')
#parser.add_argument('--val_meas_filenames', default='./casia_file_names_test1.txt')#test_flatcam_61_87.txt')
parser.add_argument('--val_meas_filenames', default='test_flatcam_61_87.txt')
parser.add_argument('--train_orig_filenames', default='./casia_file_names_train.txt')
#parser.add_argument('--val_orig_filenames', default='./casia_file_names_test1.txt')#test_flatcam_61_87.txt')
parser.add_argument('--val_orig_filenames', default='./test_flatcam_61_87.txt')
parser.add_argument('--architecture',default='UNET')
parser.add_argument('--modelRoot', default='flatnet_new')
parser.add_argument('--checkpoint', default='/content/drive/MyDrive/glow_flatcam/glow_casia_trained_latest.tar')

parser.add_argument('--valFreq', default=20,type=int)#200
parser.add_argument('--pretrain',dest='pretrain', action='store_true')
parser.add_argument("--n_flow", default=16, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--no_lu", type=bool, default=True, help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", type=bool, default=False, help="use affine coupling instead of additive")
parser.add_argument("--img_size", default=256, type=int, help="image size")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")


#lossweightage and gradientweightage
parser.add_argument('--wtp', default=1.2, type=float)
parser.add_argument('--wtmse', default=1, type=float)
parser.add_argument('--wta', default=0.6, type=float)
parser.add_argument('--generatorLR', default=1e-4, type=float)
parser.add_argument('--discriminatorLR', default=1e-4, type=float)
parser.add_argument('--init', default='Transpose')
parser.add_argument('--numEpoch', default=20,type=int)
parser.add_argument('--disPreEpochs', default=5,type=int)


parser.set_defaults(pretrain=True)

opt = parser.parse_args()

device = torch.device("cuda")
#data = '/media/data/salman/Amplitude Mask/models/'
data = './Amplitude Mask_01/models/'

d=sio.loadmat('/content/drive/MyDrive/glow_flatcam/flatnet-flatnet-sep/data/flatcam_prototype2_calibdata.mat')
phil=np.zeros((500,256,1))
phir=np.zeros((620,256,1))
phil[:,:,0]=d['P1gb']
phir[:,:,0]=d['Q1gb']
phil=phil.astype('float32')
phir=phir.astype('float32')

"""gen = FlatNet(phil,phir,4).to(device)	
vgg = Vgg16(requires_grad=False).to(device)
dis = Discriminator().to(device)"""

gen = nn.DataParallel(FlatNet(phil,phir,opt, 4)).to(device)
vgg = nn.DataParallel(Vgg16(requires_grad=False)).to(device)

dis = nn.DataParallel(Discriminator(opt)).to(device)#device_ids=[1, 2]

# gen = FlatNet(phil,phir,opt, 4).to(device)
# vgg = Vgg16(requires_grad=False).to(device)
# dis = Discriminator(opt).to(device)#device_ids=[1, 2]


gen_criterion = nn.MSELoss()
dis_criterion = nn.BCELoss()
print("hurray")
#checkpoint = os.path.join(data, opt.checkpoint)
ckpt = torch.load('/content/drive/MyDrive/glow_flatcam/latest2.tar')
#optim_gen.load_state_dict(ckpt['optimizerG_state_dict'])
#optim_dis.load_state_dict(ckpt['optimizerD_state_dict'])
dis.load_state_dict(ckpt['dis_state_dict'])
gen.load_state_dict(ckpt['gen_state_dict'])
#print('Loaded checkpoint from:'+checkpoint+'/latest.tar')
 
params_val = {'batch_size': 1,
		  'shuffle': False,
		  'num_workers': 2}

# data_dir = "/content/drive/MyDrive/FER2013/fer2013TrainValid/Fer2013TrainValid_Flatcam_measurements"
data_dir = "/content/drive/MyDrive/FER2013/fer2013TrainValid/Fer2013TrainValid_Flatcam_measurements"
print('ferplus')
#data_dir = "/content/drive/MyDrive/FER2013/Temp"
# val_dataset = DatasetFromFilenames(opt.val_meas_filenames,opt.val_orig_filenames)
val_dataset =  ImageFolderWithPaths(data_dir)
val_loader = torch.utils.data.DataLoader(val_dataset, **params_val)
wts = [opt.wtmse, opt.wtp, opt.wta]

#test(gen, dis, vgg, val_loader, val_dataset, gen_criterion, dis_criterion, './test_results_casia/', device, opt)
test(gen, dis, vgg, val_loader, val_dataset, gen_criterion, dis_criterion, '/content/drive/MyDrive/glow_flatcam/results', device, opt)



