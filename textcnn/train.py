#训练

from torch.utils.data import Dataset,DataLoader
from torch import optim
import torch 
import torch.nn as nn
import numpy as np
from model import TextCNN
from textCNN_data import textCNN_data
from config import TNNConfig
import os

def main():
	#init net
	print("init net ...")
	net = TextCNN()
	if os.path.exists(TNNConfig.weightFile):
		print("load net...")
		net.load_state_dict(torch.load(TNNConfig.weightFile))




	dataset = textCNN_data()
	dataloader = DataLoader(dataset,batch_size = TNNConfig.batch_size,shuffle = True)
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(net.parameters(),lr = TNNConfig.lr)

    #training
	for epoch in range(TNNConfig.num_epochs):
		for i ,(clas,sentences) in enumerate(dataloader):
			optimizer.zero_grad()
			sentences = sentences.type(torch.LongTensor)
			clas  = clas.type(torch.LongTensor)
			out  = net(sentences)
			loss = criterion(out,clas)
			
			loss.backward()
			optimizer.step()

			#每一个step打印一次
			print("epoch:{}\t step:{}\t loss:{}".format(epoch+1,i+1,loss.item()))

		#每次epoch更新一次模型
		print("save model....")
		torch.save(net.state_dict(),TNNConfig.weightFile)

if __name__ =="__main__":
	main()
			
			
			
	
        
            
