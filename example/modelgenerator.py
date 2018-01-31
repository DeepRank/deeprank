from deeprank.learn import *
from deeprank.learn.modelGenerator import *
conv_layers = []
conv_layers.append(conv(output_size=4,kernel_size=2,post='relu'))
conv_layers.append(pool(kernel_size=2))
conv_layers.append(conv(input_size=4,output_size=5,kernel_size=2,post='relu'))
conv_layers.append(pool(kernel_size=2))

fc_layers = []
fc_layers.append(fc(output_size=84,post='relu'))
fc_layers.append(fc(input_size=84,output_size=1))

MG = NetworkGenerator(name='cnn',fname='model.py',conv_layers=conv_layers,fc_layers=fc_layers)
MG.print()
MG.write()