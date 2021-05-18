import numpy as np
import skimage.transform
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from io import BytesIO
import torchvision.transforms as transforms
from torchvision import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
#import kornia as kr
import scipy.io 
import numpy as np
#import kornia as kr 
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision
import os
import torch
import scipy.io as sio
import numpy as np
import os
import skimage.io
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import loadmat
#import flatcam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.misc
from PIL import Image
import numpy as np
import skimage.transform
import torch
from PIL import Image
import torchvision
import torch.nn.functional as F
from io import BytesIO
import torchvision.transforms as transforms
from torchvision import datasets



mat = scipy.io.loadmat('/content/drive/MyDrive/ColabNotebooks/lensless_imaging/flatcam_calibdata.mat')
cSize = np.squeeze(mat['cSize'][:, :]).astype(int)



Plr  = torch.from_numpy(np.squeeze(mat['P1r'][:,:]).astype(float)).float()
Plgb = torch.from_numpy(np.squeeze(mat['P1gb'][:,:]).astype(float)).float()
Plgr = torch.from_numpy(np.squeeze(mat['P1gr'][:,:]).astype(float)).float()
Plb  = torch.from_numpy(np.squeeze(mat['P1b'][:,:]).astype(float)).float()
Qlr  = torch.from_numpy(np.squeeze(mat['Q1r'][:,:]).astype(float)).float()
Qlgb = torch.from_numpy(np.squeeze(mat['Q1gb'][:,:]).astype(float)).float()
Qlgr = torch.from_numpy(np.squeeze(mat['Q1gr'][:,:]).astype(float)).float()
Qlb  = torch.from_numpy(np.squeeze(mat['Q1b'][:,:]).astype(float)).float()

batch_size = 32 
Plr = torch.unsqueeze(Plr, 0).expand((batch_size, Plr.size()[0], Plr.size()[1]))
Plgb = torch.unsqueeze(Plgb, 0).expand((batch_size, Plgb.size()[0], Plgb.size()[1]))
Plgr = torch.unsqueeze(Plgr, 0).expand((batch_size, Plgr.size()[0], Plgr.size()[1]))
Plb = torch.unsqueeze(Plb, 0).expand((batch_size, Plb.size()[0], Plb.size()[1]))
Qlr = torch.unsqueeze(Qlr, 0).expand((batch_size, Qlr.size()[0], Qlr.size()[1]))
Qlgb = torch.unsqueeze(Qlgb, 0).expand((batch_size, Qlgb.size()[0], Qlgb.size()[1]))
Qlgr = torch.unsqueeze(Qlgr, 0).expand((batch_size, Qlgr.size()[0], Qlgr.size()[1]))
Qlb = torch.unsqueeze(Qlb, 0).expand((batch_size, Qlb.size()[0], Qlb.size()[1]))


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

def apply_noise(img, nSig = 10):
    r''' This function add simulated noise '''
    if nSig == 0:
        return img 
    for i in range(0, img.size()[1]):
        Y = img[i, :, :, :]
        tempY = Y - torch.min(torch.zeros(1, device = torch.device("cuda")), torch.min(Y.view(-1)))
        normY = torch.dist(tempY, torch.zeros(1, device = torch.device("cuda")), p = 2) 
        noise = torch.randn(Y.size(), device = torch.device('cuda'))
        noise = torch.sqrt((normY / nSig) ** 2 / (Y.numel() * torch.var(noise.view(-1)) )) * noise        
        img[i, :, :, :] = Y + noise

    return img 

class AddNoise(nn.Module):
    r'''Add noise for simulated measurement'''
    def __init__(self, nSig = 10):
        super(AddNoise, self).__init__()
        self.nSig = nSig

    def forward(self, x):
        return apply_noise(x, self.nSig)

class ApplyRaw2Bayer(nn.Module):
    r''' Convert Raw data to Bayer pattern'''
    def __init__(self):
        super(ApplyRaw2Bayer, self).__init__()

    def forward(self, x):
        return Raw2Bayer(x)

def Raw2Bayer(x, crop_size = cSize, is_rotate = False):
    r''' Convert FlatCam raw data to Bayer'''
    
    # Step 1. Convert the Image & rotate 
    c, b, h, w = x.size()
    
    #y = torch.zeros((c, 4, int(h/2), int(w/2)), device = torch.device('cuda'))
    y = torch.zeros((c, 4, int(h/2), int(w/2)))
    if is_rotate:                       # ---> THIS MODES DOESNOT WORK YET!!! (2019.07.14)
        scale = torch.ones(1)
        angle = torch.ones(1) * 0.05 * 360              # 0.05 is angle collected from data measurements 
        center = torch.ones(1, 2)
        center[..., 0] = int(h / 4)  # x
        center[..., 1] = int(w / 4)  # y
        M = kr.get_rotation_matrix2d(center, angle, scale).cuda()
        _, _, h, w = y.size()
        
        y[:, 0, :, : ] = kr.warp_affine(x[:, :, 1::2, 1::2], M, dsize = (h, w))
        y[:, 1, :, : ] = kr.warp_affine(x[:, :, 0::2, 1::2], M, dsize = (h, w))
        y[:, 2, :, : ] = kr.warp_affine(x[:, :, 1::2, 0::2], M, dsize = (h, w))
        y[:, 3, :, : ] = kr.warp_affine(x[:, :, 0::2, 0::2], M, dsize = (h, w))

    else:
        y[:, 0, :, : ] = x[:, 0, 1::2, 1::2]
        y[:, 1, :, : ] = x[:, 0, 0::2, 1::2]
        y[:, 2, :, : ] = x[:, 0, 1::2, 0::2]
        y[:, 3, :, : ] = x[:, 0, 0::2, 0::2]

    # Step 3. Crop the image 
    start_row = int((y.size()[2] - crop_size[0]) / 2) 
    end_row = start_row + crop_size[0]
    start_col = int((y.size()[3] - crop_size[1])/2) 
    end_col = start_col + crop_size[1] 
    return y[:,:, start_row:end_row, start_col:end_col]

def Bayer2RGB(x, normalize = True):
    b, _, h, w = x.size()
    x_rgb = torch.zeros((b, 3, h, w)).cuda()    
    x_rgb[:, 0, :, :] = x[:, 0, :, :]
    x_rgb[:, 1, :, :] = 0.5 * (x[:, 1, :, :]  + x[:, 2, :, :])
    x_rgb[:, 2, :, :] = x[:, 3, :, :]

    if normalize:
        x_rgb = (x_rgb - torch.min(x_rgb.view(-1))) / (torch.max(x_rgb.view(-1)) - torch.min(x_rgb.view(-1)) )
    
    return x_rgb 

class ApplyBayer2RGB(nn.Module):
    def __init__(self, normalize = True ):
        super(ApplyBayer2RGB, self).__init__()
        self.normalize = normalize
    
    def forward(self, x):
        return Bayer2RGB(x, self.normalize)
    

def flatcamSamp(x):
    y = torch.zeros((x.size()[0], 1, Plr.size()[1] * 2, 2 * Qlr.size()[1])).to(torch.device("cuda"))       
    y[:, 0, 1::2, 1::2] = torch.bmm(torch.bmm(Plr[0:x.size()[0], :, :].cuda(),  x[:, 0, :, :].cuda()), Qlr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 0::2, 1::2] = torch.bmm(torch.bmm(Plgb[0:x.size()[0], :, :].cuda(), x[:, 1, :, :].cuda()), Qlgb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 1::2, 0::2] = torch.bmm(torch.bmm(Plgr[0:x.size()[0], :, :].cuda(), x[:, 1, :, :].cuda()), Qlgr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))
    y[:, 0, 0::2, 0::2] = torch.bmm(torch.bmm(Plb[0:x.size()[0], :, :].cuda() , x[:, 2, :, :].cuda()), Qlb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]))

    return y 
class FlatCamSampSim(nn.Module):
    r''' Simulated Flatcam measurement '''
    def __init__(self, batSize):
        super(FlatCamSampSim, self).__init__()
        if batSize > batch_size:
            raise Exception('batch_size should not exceed {}. Please change the corresponding batch_size values in common.py file'.format(batch_size))      

    def forward(self, x):        
        return flatcamSamp(x)

class FlatCamSimInverse(nn.Module):
    r''' Initial Reconstruction for Simulated'''
    def __init__(self):
        super(FlatCamSimInverse, self).__init__()           
        
    def forward(self, x):
        # Step 0 Convert from raw data to bayer 
        x = Raw2Bayer(x)
        
        # Step 2: Simple Inverse 
        y = torch.zeros((x.size()[0], 4, Plr.size()[2], Qlr.size()[2])).to(torch.device("cuda"))
        y[:, 0, :, :] = torch.bmm(torch.bmm(Plr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]),  x[:, 0, :, :]), Qlr[0:x.size()[0], :, :].cuda())        
        y[:, 1, :, :] = torch.bmm(torch.bmm(Plgb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]), x[:, 1, :, :].cuda()), Qlgb[0:x.size()[0], :, :].cuda())
        y[:, 2, :, :] = torch.bmm(torch.bmm(Plgr[0:x.size()[0], :, :].cuda().permute([0, 2, 1]), x[:, 1, :, :].cuda()), Qlgr[0:x.size()[0], :, :].cuda())
        y[:, 3, :, :] = torch.bmm(torch.bmm(Plb[0:x.size()[0], :, :].cuda().permute([0, 2, 1]),  x[:, 2, :, :].cuda()), Qlb[0:x.size()[0], :, :].cuda())

        # Step 3: Convert to bayer pattern 
        y = F.relu(y)               # Remove negative value  --> maybe not necessary 
        y = Bayer2RGB(y)            # convert to RGB 

        return y 

def make_separable(x):

    ''' function that convert separable of image'''
    #b, c, w, h = x.size() 

    #for i in range(b):
    rowMeans = torch.mean(x, 3)
    colMeans = torch.mean(x, 2) 
    allMean = torch.mean(rowMeans, 2)

    rowMeans = torch.unsqueeze(rowMeans, -1).expand(x.size())
    colMeans = torch.unsqueeze(colMeans, 2).expand(x.size())
    allMean = torch.unsqueeze(torch.unsqueeze(allMean, -1), -2).expand(x.size())

    x = x - rowMeans - colMeans + allMean
    
    return x




def demosaic_raw(meas):
    # inv = FlatCamSimInverse()
    # recon = inv(img.unsqueeze(0))
    meas = Raw2Bayer(meas.unsqueeze(0))
    meas = meas.squeeze(0)
    
# 	tform = skimage.transform.SimilarityTransform(rotation=0.00174)
# 	X = meas.numpy()[0,:,:]
# 	#print(f'sairam2 ---> {X.shape}')
# 	X = X/65535.0
# 	X=X+0.003*np.random.randn(X.shape[0],X.shape[1])
# 	im1=np.zeros((512,640,4))
# 	im1[:,:,0]=X[0::2, 0::2]#b
# 	im1[:,:,1]=X[0::2, 1::2]#gb
# 	im1[:,:,2]=X[1::2, 0::2]#gr
# 	im1[:,:,3]=X[1::2, 1::2]#r
# 	im1=skimage.transform.warp(im1,tform)
# 	im=im1[6:506,10:630,:]      
# 	rowMeans = im.mean(axis=1, keepdims=True)
# 	colMeans = im.mean(axis=0, keepdims=True)
# 	allMean = rowMeans.mean()
# 	im = im - rowMeans - colMeans + allMean # this looks wrong; first the tensor should be permuted and then the means should be computed across h and w
# 	im = im.astype('float32')
# 	#meas = torch.from_numpy(np.swapaxes(np.swapaxes(im,0,2),1,2)).unsqueeze(0)
# 	meas = torch.from_numpy(np.expand_dims(im, axis=0)).permute(0, 3, 1, 2)
    return meas
	#return meas[0,:,:,:]

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        meas, _ = super(ImageFolderWithPaths, self).__getitem__(index)
        path,_ = self.imgs[index]
        image_name = os.path.basename(path)
        #y = torch.randn(32,3,256,256)
        #image_name = os.path.basename(path)
        meas = transforms.ToTensor()(meas)
        print(f'sairam1 ---> {meas.size()}')
        meas = demosaic_raw(meas)
        print(f'sairam3 ---> {meas.size()}')
        #meas = transforms.ToTensor()(meas)
        return meas,image_name

# class DatasetFromFilenames:

#   def __init__(self, filenames_loc_meas, filenames_loc_orig):
#     self.filenames_meas = filenames_loc_meas
#     #self.paths_meas = get_paths(self.filenames_meas, "/home/darshan/data/ug2challengedata/casia/lensless/new_measurements/")
#     #self.paths_meas = get_paths(self.filenames_meas, "/content/drive/MyDrive/RAFDB_Alligned/measurements")
    
#     self.paths_meas = get_paths(self.filenames_meas, "/content/drive/MyDrive/RAFDB_Alligned/measurements")
    
#     self.filenames_orig = filenames_loc_orig
#     #self.paths_orig = get_paths(self.filenames_orig, "/home/darshan/data/ug2challengedata/casia/lens/images/")
#     self.paths_orig = get_paths(self.filenames_orig, "/content/drive/MyDrive/RAFDB_Alligned/Aligned/aligned")
#     self.num_im = len(self.paths_meas)
#     self.totensor = torchvision.transforms.ToTensor()
#     self.resize = torchvision.transforms.Resize((256,256))
#   def __len__(self):
#     return len(self.paths_meas)

#   def __getitem__(self, index):
#     im_path = self.paths_orig[index % self.num_im]
#     meas_path = self.paths_meas[index % self.num_im]
#     # load images (grayscale for direct inference)
#     im = Image.open(im_path)
#     im = im.convert('RGB')
#     im = self.resize(im)
#     # print(im.size)
#     im = self.totensor(im)
    
#     meas = Image.open(meas_path)
#     meas = self.totensor(meas)
#     # print(meas.shape)
#     meas = demosaic_raw(meas)
#     return meas,im


# #def get_paths(fname):
# def get_paths(fname, root):
#   paths = []
#   if "images" in root:
#     ext = ".png"#'.png' for flatcam lens images  # for casia ".jpg"
#   else:
#     ext = ".png"
#   with open(fname, 'r') as f:
#     for line in f:
#       temp = root+str(line).strip().split(' ')[0]+ext
#       paths.append(temp)
#   return paths
"""
def get_paths_test(fname, root):
	paths = []
	if "images" in root:
		ext = ".jpg"
	else:
		ext = ".png"
    
	with open(fname, 'r') as f:
		for line in f:
			#temp = '/media/data/salman/'+str(line).strip()
			temp = root+str(line).strip()+ext
			paths.append(temp)
	return paths
"""


if __name__ == "__main__":
	d = DatasetFromFilenames("casia_file_names_test.txt", "casia_file_names_test.txt")
	print(len(d))
	d_iter = iter(d)
	print(next(d_iter)[0])

