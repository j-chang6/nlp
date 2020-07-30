#进行测试

import numpy as np
import torch
import os
from config import TNNConfig
from data_loader import *
from model import TextCNN
import random



def main():
	print("init net...")
	net = TextCNN()
	if os.path.exists(TNNConfig.weightFile):
		print("load weight")
		net.load_state_dict(torch.load(TNNConfig.weightFile))
	else:
		print("no weight file")
		exit()
	
	net.eval()
	#获得testdata，数据元素是numerical的seq[seq_len] 
	testData = txt2ind(TNNConfig.myTrainFile)
	random.shuffle(testData)
	
	num_all = 0
	num_right = 0
	for data in testData:
		label = int(data[0])
		sentence = np.array([int(x) for x in data[1:]])
		sentence = torch.from_numpy(sentence)
		
		#模型的输入数据为[batch,seq_len]，所以在进入之前需要进行unsqueeze(0)
		#变为[1,seq_len]
		#数据变为numpy()，需要detach()来切断反向传播
		predict = net(sentence.unsqueeze(0).type(torch.LongTensor)).detach().numpy()[0]
		out = np.argmax(predict)
		score = max(predict)
		
		if out == label and score > 0 :
			num_right +=1
		num_all +=1
	
	print("acc:{:.2f}({}/{})".format(num_right/num_all,num_right,num_all))


if __name__ =="__main__":
	main()