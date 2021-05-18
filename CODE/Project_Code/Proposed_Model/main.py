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
from models import*
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
from tqdm import tqdm

#CUDA_LAUNCH_BLOCKING=1

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2"

parser = argparse.ArgumentParser()
#model and data locs
parser.add_argument('--train_meas_filenames', default='./casia_file_names_train.txt')
parser.add_argument('--val_meas_filenames', default='./casia_file_names_test.txt')
parser.add_argument('--train_orig_filenames', default='./casia_file_names_train.txt')
parser.add_argument('--val_orig_filenames', default='./casia_file_names_test.txt')

parser.add_argument('--modelRoot', default='flatnet_new')
parser.add_argument('--checkpoint', default='')
#lossweightage and gradientweightage
parser.add_argument('--wtp', default=1.2, type=float)
parser.add_argument('--wtmse', default=1, type=float)
parser.add_argument('--wta', default=0.6, type=float)
parser.add_argument('--wttex', type=float, default=1, help='texture loss wt. Default=1')
#LR
parser.add_argument('--generatorLR', default=1e-4, type=float)
parser.add_argument('--discriminatorLR', default=1e-4, type=float)
#others
parser.add_argument('--init', default='Transpose')

parser.add_argument('--valFreq', default=20,type=int)#200
parser.add_argument('--pretrain',dest='pretrain', action='store_true')
parser.add_argument("--n_flow", default=16, type=int, help="number of flows in each block")
parser.add_argument("--n_block", default=4, type=int, help="number of blocks")
parser.add_argument("--no_lu", type=bool, default=True, help="use plain convolution instead of LU decomposed version")
parser.add_argument("--affine", type=bool, default=False, help="use affine coupling instead of additive")
parser.add_argument("--img_size", default=256, type=int, help="image size")
parser.add_argument("--n_bits", default=5, type=int, help="number of bits")
parser.add_argument("--batch_size", default=24, type=int, help="batch size")

parser.add_argument("--pretrain_disc_iter", default=2100, type=int)#35 for 1000
parser.add_argument("--iter", default=10000, type=int)#7000 for 1000
parser.add_argument("--lr_period", default=1000, type=int)

parser.set_defaults(pretrain=True)

opt = parser.parse_args()
device = torch.device("cuda")
data = './Amplitude Mask/models/'
savedir = os.path.join(data, opt.modelRoot)
class Logger(object):
	def __init__(self, save_dir):
		self.terminal = sys.stdout
		self.log = open(os.path.join(save_dir, "log.txt"), "a+")

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		self.terminal.flush()
if not os.path.exists(savedir):
	os.mkdir(savedir)
sys.stdout = Logger(savedir)
print('======== Log ========')
print(datetime.now(timezone('Asia/Kolkata')))
print('\n')
print("Command ran:\n%s\n\n" % " ".join([x for x in sys.argv]))
print("Opt:")
pprint.pprint(vars(opt))
print("\n")

vla = float('inf')
k = 0
val_err = []
train_err = []
sys.stdout.flush()

if opt.init=='Transpose':
	print('Loading calibrated files')
	d=sio.loadmat('data/flatcam_prototype2_calibdata.mat')
	phil=np.zeros((500,256,1))
	phir=np.zeros((620,256,1))
	phil[:,:,0]=d['P1gb']
	phir[:,:,0]=d['Q1gb']
	phil=phil.astype('float32')
	phir=phir.astype('float32')
else:
	print('Loading Random Toeplitz')
	phil=np.zeros((500,256,1))
	phir=np.zeros((620,256,1))
	pl = sio.loadmat('data/phil_toep_slope22.mat')
	pr = sio.loadmat('data/phir_toep_slope22.mat')
	phil[:,:,0] = pl['phil'][:,:,0]
	phir[:,:,0] = pr['phir'][:,:,0]
	phil=phil.astype('float32')
	phir=phir.astype('float32')

gen = FlatNet(phil,phir,opt,4)	
vgg = Vgg16(requires_grad=False)
dis = Discriminator(opt)

gen = nn.DataParallel(gen, device_ids=[0, 1, 2]).to(device)	
vgg = nn.DataParallel(vgg, device_ids=[0, 1, 2]).to(device)
dis = nn.DataParallel(dis, device_ids=[0, 1, 2]).to(device)

gen_criterion = nn.MSELoss()#nn.L1Loss()#
dis_criterion = nn.BCELoss()




ei = 0
train_error = []
val_error = []


optim_gen = torch.optim.Adam(gen.parameters(), lr = opt.generatorLR)
optim_dis = torch.optim.Adam(dis.parameters(), lr = opt.discriminatorLR)

vla = float('inf')
if opt.checkpoint:
	checkpoint = os.path.join(data, opt.checkpoint)
	ckpt = torch.load(checkpoint+'/latest.tar')
	optim_gen.load_state_dict(ckpt['optimizerG_state_dict'])
	optim_dis.load_state_dict(ckpt['optimizerD_state_dict'])
	dis.load_state_dict(ckpt['dis_state_dict'])
	gen.load_state_dict(ckpt['gen_state_dict'])
	ei = ckpt['last_finished_epoch'] + 1
	val_error = ckpt['val_err']
	train_error = ckpt['train_err']
	vla = min(ckpt['val_err'])
	print('Loaded checkpoint from:'+checkpoint+'/latest.tar')

for param_group in optim_gen.param_groups:
	genLR = param_group['lr']
for param_group in optim_dis.param_groups:
	disLR = param_group['lr']
 
 
params_train = {'batch_size': 24,
		  'shuffle': True,
		  'num_workers': 8}

params_val = {'batch_size': 1,
		  'shuffle': False,
		  'num_workers': 1}
train_dataset = DatasetFromFilenames(opt.train_meas_filenames,opt.train_orig_filenames)
val_dataset = DatasetFromFilenames(opt.val_meas_filenames,opt.val_orig_filenames)
train_loader = torch.utils.data.DataLoader(train_dataset, **params_train)
val_loader = torch.utils.data.DataLoader(val_dataset, **params_val)


wts = [opt.wtmse, opt.wtp, opt.wta, opt.wttex]

disc_err = []
"""if opt.pretrain and not opt.checkpoint:
	disc_err = train_discriminator_epoch(gen, dis, optim_dis, dis_criterion, train_loader, train_dataset, disc_err, device, opt)
torch.save(dis.state_dict(), savedir+'/pretrained_disc_24_noaffine.tar')
torch.save(optim_dis.state_dict(), savedir+'/pretrained_optim_disc_24_noaffine.tar')
assert False"""
# load pre-trained disc
checkpoint = os.path.join(data, opt.checkpoint)
dis.load_state_dict(torch.load(checkpoint+'flatnet_new/pretrained_disc_24_noaffine.tar'))
optim_dis.load_state_dict(torch.load(checkpoint+'flatnet_new/pretrained_optim_disc_24_noaffine.tar'))

params_train = {'batch_size': opt.batch_size,
		  'shuffle': True,
		  'num_workers': 8}
train_loader = torch.utils.data.DataLoader(train_dataset, **params_train)
train_full_epoch(gen, dis, vgg, wts, optim_gen, optim_dis,	train_loader, train_dataset, val_loader, val_dataset, gen_criterion, dis_criterion, device, vla,  savedir, train_error, val_error, disc_err, sys.stdout,opt.valFreq, opt)


#for e in range(ei,opt.numEpoch):
#for e in tqdm(range(ei,opt.numEpoch)):
   
#sys.stdout.flush()
   
"""Xvalout = Xvalout.cpu()
ims = Xvalout.detach().numpy()
ims = ims[0, :, :, :]
ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
skimage.io.imsave(savedir+'/latest.png', ims)

dict_save = {
		'gen_state_dict': gen.state_dict(),
		'dis_state_dict': dis.state_dict(),
		'optimizerG_state_dict': optim_gen.state_dict(),
		'optimizerD_state_dict': optim_dis.state_dict(),
		'train_err': train_error,
		'val_err': val_error,
		'disc_err': disc_err,
		'last_finished_epoch': e,
		'opt': opt,
		'vla': vla}
torch.save(dict_save, savedir+'/latest.tar')
savename = '/phil_epoch%d' % e
np.save(savedir+savename, gen.module.PhiL.detach().cpu().numpy())
savename = '/phir_epoch%d' % e
np.save(savedir+savename, gen.module.PhiR.detach().cpu().numpy())
if e%2 == 0:
	genLR = genLR/2
	disLR = disLR/2
	for param_group in optim_gen.param_groups:
		param_group['lr'] = genLR
	for param_group in optim_dis.param_groups:
		param_group['lr'] = disLR

print('Saved latest')
sys.stdout.flush()"""
