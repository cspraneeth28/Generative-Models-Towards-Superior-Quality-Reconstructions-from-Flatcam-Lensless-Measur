import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from glow_model import Glow

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)
def dilconv3x3(in_channels, out_channels, stride=1,dilation=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, dilation=dilation, padding=2, bias=False)
def Bayer2RGB(x, normalize = True):
    b, _, h, w = x.size()
    x_rgb = torch.zeros((b, 3, h, w)).cuda()#to(device)
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


class double_conv(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x

	
	
class double_conv2(nn.Module):
	'''(conv => BN => ReLU) * 2'''
	def __init__(self, in_ch, out_ch):
		super(double_conv2, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(in_ch, out_ch, 3,stride=2, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True),
			nn.Conv2d(out_ch, out_ch, 3, padding=1),
			nn.BatchNorm2d(out_ch,momentum=0.99),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		x = self.conv(x)
		return x    


class inconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(inconv, self).__init__()
		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.conv(x)
		return x


class down(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(down, self).__init__()
		self.mpconv = nn.Sequential(
			double_conv2(in_ch, out_ch)
		)

	def forward(self, x):
		x = self.mpconv(x)
		return x


class up(nn.Module):
	def __init__(self, in_ch, out_ch, bilinear=False):
		super(up, self).__init__()

		if bilinear:
			self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
		else:
			self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x1, x2):
		x1 = self.up(x1)
		diffX = x1.size()[2] - x2.size()[2]
		diffY = x1.size()[3] - x2.size()[3]
		x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
						diffY // 2, int(diffY / 2)))
		x = torch.cat([x2, x1], dim=1)
		x = self.conv(x)
		return x


class upnocat(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(upnocat, self).__init__()

		self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

		self.conv = double_conv(in_ch, out_ch)

	def forward(self, x):
		x = self.up(x)
		x = self.conv(x)
		return x


class outconv(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(outconv, self).__init__()
		self.conv = nn.Conv2d(in_ch, out_ch, 3,padding=1)

	def forward(self, x):
		x = self.conv(x)
		return x


# In[7]:


def swish(x):
	return x * torch.sigmoid(x)

class FlatNet(nn.Module):
	def __init__(self, phil, phir, args, n_channels=4):
		super(FlatNet, self).__init__()
		self.args = args
		self.PhiL =nn.Parameter(torch.tensor(phil)) 
		self.PhiR=nn.Parameter(torch.tensor(phir)) 
		self.bayer_to_RGB = ApplyBayer2RGB(normalize=True)
		#self.bn=nn.BatchNorm2d(3,momentum=0.99)
		self.gen = Glow(3, args.n_flow, args.n_block, affine=args.affine, conv_lu=True)
		
	def forward(self, Xinp):
		
		X0=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,0,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X11=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,1,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X12=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,2,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		X2=F.leaky_relu(torch.matmul(torch.matmul(Xinp[:,3,:,:],self.PhiR[:,:,0]).permute(0,2,1),self.PhiL[:,:,0]).permute(0,2,1).unsqueeze(3))
		Xout=torch.cat((X2,X12,X11,X0),3)
		x = Xout.permute(0,3,1,2)

		x = self.bayer_to_RGB(x)
		flatcam_sim = x
		"""temp1 = torch.amin(x, dim=(2, 3), keepdims=True)
		temp2 = torch.amax(x, dim=(2, 3), keepdims=True)
		x = (x-temp1)/(temp2-temp1)#self.bn(x)"""	
		x = x * 255
		if self.args.n_bits <= 8:
			x = torch.floor(x / 2 ** (8 - self.args.n_bits))
		x = x / (2**self.args.n_bits) - 0.5	
		x = x + torch.rand_like(x) / (2**self.args.n_bits)

		log_p, log_det, z_outs, means = self.gen(x)
		generated = self.gen(z_outs, reverse=True)

		#print("generated before...........\n\n", generated.min(), generated.max())
		#generated.clamp_(-2., 2.)
		print("generated after...........", generated.min(), generated.max())
		

		return torch.sigmoid(generated), Xout, log_p, log_det, z_outs, means,flatcam_sim

class TikNet(nn.Module):
	## This model is basically U-Net. It was used to train the naive model on Tikhonov reconstructions.
    def __init__(self, n_channels):
        super(TikNet, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 1024)
        self.down4 = down(1024, 1024)
        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 128)
        self.outc = outconv(128, 3)
        self.bn=nn.BatchNorm2d(3,momentum=0.99)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x),x5

class Discriminator(nn.Module):
	def __init__(self, args):
		self.args = args
		super(Discriminator, self).__init__()
		self.conv2 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv4 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(64)
		self.conv6 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
		self.bn6 = nn.BatchNorm2d(128)
		self.conv9 = nn.Conv2d(128, 1, 1, stride=1, padding=1)

	def forward(self, x):

		x = x * 255
		if self.args.n_bits <= 8:
			x = torch.floor(x / 2 ** (8 - self.args.n_bits))
		x = x / (2**self.args.n_bits) - 0.5
		
		x = swish(self.bn2(self.conv2(x)))
		x = swish(self.bn4(self.conv4(x)))
		x = swish(self.bn6(self.conv6(x)))
		x = self.conv9(x)
		return torch.sigmoid(F.avg_pool2d(x, x.size()[2:])).view(x.size()[0], -1)




