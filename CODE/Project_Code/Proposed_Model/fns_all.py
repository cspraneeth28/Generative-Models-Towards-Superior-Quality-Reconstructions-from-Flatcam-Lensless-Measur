import torch
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.color import rgb2gray
import skimage.io
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from skimage.transform import resize as rsz
import torch.optim as optim
import os
from models import*
import torchvision
#from torch_vgg import Vgg16

from math import log, sqrt, pi
from tqdm import tqdm
from matplotlib.lines import Line2D

def plot_grad_flow(named_parameters):
	'''Plots the gradients flowing through different layers in the net during training.
	Can be used for checking for possible gradient vanishing / exploding problems.

	Usage: Plug this function in Trainer class after loss.backwards() as 
	"plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
	ave_grads = []
	max_grads= []
	layers = []
	for n, p in named_parameters:
		if(p.requires_grad) and ("bias" not in n):
		    layers.append(n)
		    ave_grads.append(p.grad.abs().mean().cpu())
		    max_grads.append(p.grad.abs().max().cpu())
	plt.bar(np.arange(len(max_grads[:20])), max_grads[:20], alpha=0.1, lw=1, color="c")
	plt.bar(np.arange(len(max_grads[:20])), ave_grads[:20], alpha=0.1, lw=1, color="b")
	plt.hlines(0, 0, len(ave_grads[:20])+1, lw=2, color="k" )
	plt.xticks(range(0,len(ave_grads[:20]), 1), layers[:20], rotation="vertical")
	plt.xlim(left=0, right=len(ave_grads[:20]))
	plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
	plt.xlabel("Layers")
	plt.ylabel("average gradient")
	plt.title("Gradient flow")
	plt.grid(True)
	plt.legend([Line2D([0], [0], color="c", lw=4),
		        Line2D([0], [0], color="b", lw=4),
		        Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
	plt.show()


def sample_data(args, dataset, loader):
	loader = iter(loader)
	while True:
		try:
			yield next(loader)
		except StopIteration:
			loader = DataLoader(
				dataset, shuffle=True, batch_size=args.batch_size, num_workers=4
			)
			loader = iter(loader)
			yield next(loader)

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

def texture_criterion(a, b):
    return torch.mean(torch.abs((a-b)**2).view(-1))
    
def calc_nll_loss(log_p, logdet, image_size, n_bins):
  n_pixel = image_size * image_size * 3
  loss = -log(n_bins) * n_pixel
  loss = loss + logdet + log_p
  return (
		(-loss / (log(2) * n_pixel)).mean(),
		(log_p / (log(2) * n_pixel)).mean(),
		(logdet / (log(2) * n_pixel)).mean(),
	)

#make train within train for gen and dis
def train_discriminator_epoch(gen, dis, optim_dis, criterion, train_loader, train_dataset, disc_err, device, args):
  print('\nPre-Training dicriminator\n')
  for param in gen.parameters():
  	param.requires_grad = False
  for param in dis.parameters():
  	param.requires_grad = True
  
  d = iter(sample_data(args, train_dataset, train_loader))
  with tqdm(range(args.pretrain_disc_iter)) as pbar:
    for i in pbar:
    	X_train, Y_train = next(d)			
    	batchsize = X_train.shape[0]
    	target_real = Variable(torch.rand(batchsize,1)*0.5 + 0.7).to(device)
    	target_fake = Variable(torch.rand(batchsize,1)*0.3).to(device)
    	X_train, Y_train = X_train.to(device), Y_train.to(device)
    	optim_dis.zero_grad()
    	dis.train()
    	high_res_real = Variable(Y_train)
    	high_res_fake,_,log_p, log_det, z_outs, means = gen(X_train)
    	dis_loss = criterion(dis(high_res_real), target_real) + criterion(dis(Variable(high_res_fake.data)), target_fake)
    	dis_loss.backward()
    	optim_dis.step()
    	disc_err.append(dis_loss.item())			
  return disc_err



def test(gen, dis, vgg, val_loader, val_dataset, gen_criterion, dis_criterion, savedir, device, args): #BY us for all images saving
	gen.eval()
	with torch.no_grad():
		for i, (X_val, Y_val) in enumerate(val_loader):						
			batchsize = X_val.shape[0]
			ones_const = Variable(torch.ones(batchsize, 1)).to(device)
			# X_val, Y_val = batchGenerator(1, arr[i], h, phase_mask_fwd,device,pathstr)
			#X_val, Y_val = X_val.to(device), Y_val.to(device)
			X_val = X_val.to(device)
			# print(Y_val.shape)
			X_valout,_,log_p, log_det, z_outs, means,flatcam_sim = gen(X_val)
			#print(flatcam_sim.size())
			#torchvision.utils.save_image(flatcam_sim.cpu().squeeze(0),f'/content/drive/MyDrive/FER2013/Temp/temp_fc/{Y_val[0]}')
			ims = X_valout.cpu().detach().numpy()
			ims = ims[0, :, :, :]
			ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
			ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
			skimage.io.imsave(f'/content/drive/MyDrive/glow_flatcam/Enhanced_FERPLUS_celeba_pre_trained_final/{Y_val[0]}', ims)
			print('sairaaaaaaaaazzzzzzzm')
			#op = X_valout
			
	return #op, tloss

def validate(gen, dis, vgg, wts, val_loader, val_dataset, gen_criterion, dis_criterion,device, args):
	k = 0
	tloss = 0
	gen.eval()
	with torch.no_grad():
		for X_val, Y_val in val_loader:			
			batchsize = X_val.shape[0]
			ones_const = Variable(torch.ones(batchsize, 1)).to(device)
			# X_val, Y_val = batchGenerator(1, arr[i], h, phase_mask_fwd,device,pathstr)
			X_val, Y_val = X_val.to(device), Y_val.to(device)
			# print(Y_val.shape)
			X_valout,_,log_p, log_det, z_outs, means = gen(X_val)
			valfeatures_y, _ = vgg.module(Y_val)
			valfeatures_x, _ = vgg.module(X_valout)
			#if k == 5:
			op = X_valout	
			nll_loss = 1e-3*calc_nll_loss(log_p, log_det, args.img_size, 2**args.n_bits)[0]					
			tloss = tloss + nll_loss + (wts[0]*(gen_criterion(Y_val, X_valout)+(wts[1]*gen_criterion(valfeatures_y.relu2_2, valfeatures_x.relu2_2))+(wts[1]*gen_criterion(valfeatures_y.relu4_3, valfeatures_x.relu4_3)))+wts[2]*dis_criterion(dis(X_valout), ones_const)).item()
			k += 1
		tloss = tloss/len(val_loader)
	return op, tloss


def train_full_epoch(gen, dis, vgg, wts, optim_gen, optim_dis, train_loader, train_dataset, val_loader, val_dataset,
gen_criterion, dis_criterion, device, vla,  savedir, train_error, val_error, disc_err,ss,valFreq, args):
	print('\n Training Generative model') 	
	d = iter(sample_data(args, train_dataset, train_loader))	
  
	with tqdm(range(args.iter)) as pbar:
		for i in pbar:
			if i>0 and i%args.lr_period == 0:
				print('\nlr changed.')      
				for param_group in optim_gen.param_groups:
		  			param_group['lr'] /= 2.
				for param_group in optim_dis.param_groups:
					param_group['lr'] /= 2.
	  
			X_train, Y_train = next(d)
			X_train = X_train.to(device)
			Y_train = Y_train.to(device)
			batchsize = X_train.shape[0]
			text_loss = []
      
			#Train discriminator
			ones_const = Variable(torch.ones(batchsize, 1)).to(device)

			target_real = Variable(torch.rand(batchsize,1)*0.5 + 0.7).to(device)
			target_fake = Variable(torch.rand(batchsize,1)*0.3).to(device)

			for param in gen.parameters():
				param.requires_grad = False

			for param in dis.parameters():
				param.requires_grad = True

			optim_dis.zero_grad()
			dis.train()

			high_res_real = Variable(Y_train)
			high_res_fake,_,log_p, log_det, z_outs, means = gen(X_train)
			#print("Nannnnnn", torch.any(high_res_fake.isnan()))
			dis_loss = dis_criterion(dis(high_res_real), target_real) + dis_criterion(dis(Variable(high_res_fake.data)), target_fake)
			#print("Dis_losssssss",dis_loss.item())
			dis_loss.backward()
			optim_dis.step()
			disc_err.append(dis_loss.item())

			#Train generator
			for param in gen.parameters():
				param.requires_grad = True

			for param in dis.parameters():
				param.requires_grad = False

			optim_gen.zero_grad()
			gen.train()			
			Xout,_,log_p, log_det, z_outs, means = gen(X_train)	
			features_y, texture_y = vgg.module(Y_train)
			features_x, texture_x= vgg.module(Xout)      
			nll_loss = 1e-3*calc_nll_loss(log_p, log_det, args.img_size, 2**args.n_bits)[0]
			gen_mse_loss = gen_criterion(Y_train, Xout)
			percept_loss = gen_criterion(features_y.relu2_2, features_x.relu2_2) + gen_criterion(features_y.relu4_3, features_x.relu4_3)      
			gram_x = [gram_matrix(y) for y in texture_x]
			gram_y = [gram_matrix(y) for y in texture_y]

			for m in range(0, len(gram_x)):
				text_loss += [texture_criterion(gram_x[m], gram_y[m])]
			text_loss = sum(text_loss)
           
			adv_loss = dis_criterion(dis(Xout), ones_const)	
			loss = nll_loss + wts[0]*gen_mse_loss+wts[1]*percept_loss+wts[2]*adv_loss + wts[3] * text_loss
			#loss = wts[0]*gen_mse_loss+wts[1]*percept_loss+wts[2]*adv_loss + wts[3] * text_loss  
			loss.backward()

			"""if i>0 and i%10==0:
				plot_grad_flow(gen.named_parameters())"""

			optim_gen.step()
			train_error.append(loss.item())
			if i>0 and i % valFreq == 0:
				Xvalout, vloss= validate(gen, dis, vgg, wts, val_loader, val_dataset, gen_criterion, dis_criterion, device, args)
				val_error.append(vloss)
				if vloss < vla:
					vla = vloss
					Xvalout = Xvalout.cpu()
					ims = Xvalout.detach().numpy()
					ims = ims[0, :, :, :]
					ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
					ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
					skimage.io.imsave(savedir+'/best.png', ims)
					dict_save = {
					'gen_state_dict': gen.state_dict(),
					'dis_state_dict': dis.state_dict(),
					'optimizerG_state_dict': optim_gen.state_dict(),
					'optimizerD_state_dict': optim_dis.state_dict(),
					'train_err': train_error,
					'val_err': val_error,
					'disc_err': disc_err,
					'last_finished_iter': i}
					torch.save(dict_save, savedir+'/best.tar')
					print('Saved best')           
				Xvalout = Xvalout.cpu()
				ims = Xvalout.detach().numpy()
				ims = ims[0, :, :, :]
				ims = np.swapaxes(np.swapaxes(ims,0,2),0,1)
				ims = (ims-np.min(ims))/(np.max(ims)-np.min(ims))
				skimage.io.imsave(savedir+'/'+str(i)+'_latest.png', ims)             
				dict_save = {
		  		'gen_state_dict': gen.state_dict(),
		  		'dis_state_dict': dis.state_dict(),
		  		'optimizerG_state_dict': optim_gen.state_dict(),
		  		'optimizerD_state_dict': optim_dis.state_dict(),
		  		'train_err': train_error,
		  		'val_err': val_error,
		  		'disc_err': disc_err,
		  		'last_finished_iter': i,
		  		'opt': args,
		  		'vla': vla}
				torch.save(dict_save, savedir+'/latest.tar')    

				pbar.set_description(f"disc_loss: {dis_loss:.5f}, nll_loss: {nll_loss:.5f}, gen_mse_loss: {gen_mse_loss:.5f}, percept_loss: {percept_loss:.5f}, adv_loss: {adv_loss:.5f}")
				#pbar.set_description(f"disc_loss: {dis_loss:.5f}, gen_mse_loss: {gen_L1_loss:.5f}, percept_loss: {percept_loss:.5f}, adv_loss: {adv_loss:.5f}")

				ss.flush()  
	return
			







