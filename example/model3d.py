
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils



'''
definition of the Convolutional Networks

The model must take as an argument the input shape
This allows to automaically precompute
the input size of the first FC hidden layer

'''

class cnn(nn.Module):

	def __init__(self,input_shape):
		super(cnn,self).__init__()

		self.conv1 = nn.Conv3d(input_shape[0],4,kernel_size=2)
		self.pool  = nn.MaxPool3d((2,2,2))
		self.conv2 = nn.Conv3d(4,5,kernel_size=2)
		self.conv2_drop = nn.Dropout3d()

		size = self._get_conv_output(input_shape)

		self.fc1   = nn.Linear(size,84)
		self.fc2   = nn.Linear(84,1)

		self.sm = nn.Softmax()

	def _get_conv_output(self,shape):
		inp = Variable(torch.rand(1,*shape))
		out = self._forward_features(inp)
		return out.data.view(1,-1).size(1)

	def _forward_features(self,x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		return x

	def forward(self,x):
		
		x = self._forward_features(x)
		x = x.view(x.size(0),-1)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x




