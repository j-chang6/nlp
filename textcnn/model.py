#构建cnn模型

import torch
import numpy as np
import torch.nn as nn
from config import TNNConfig

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        ci = 1  #input chanel
        kernel_num = TNNConfig.num_fifters
        kernel_size = TNNConfig.filter_sizes
        vocab_size = TNNConfig.vocab_size
        embed_dim = TNNConfig.embedding
        dropout = TNNConfig.dropout_keep
        class_num = TNNConfig.num_class
        embedding_matrix=np.load(TNNConfig.pre_w2v_npy)
        
        # self.param = param
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx= 0)
        embedding_matrix = np.array(embedding_matrix)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = True

        #进入卷积之前[batch_size,1,seq_len,embed_dim]
        self.conv = nn.Sequential(
            nn.Conv2d(ci, kernel_num, (kernel_size[0], embed_dim)),
            #卷积后[batch_size,kernel_num,H_out,1]
            nn.ReLU(),
            nn.MaxPool2d((TNNConfig.seq_len-kernel_size[0]+1,1)),
            #pool之后[batch_size,kernel_num,1,1]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num,class_num)
        


    def forward(self,X):
        #x:[batch_size,sequence_length]
        batch_size = X.shape[0]
        embedding_x = self.embedding(X) #[batch_size,seq_len,word_dim]
        embedding_x = embedding_x.unsqueeze(1) #add channel==1
        conved = self.conv(embedding_x)
        #view是reshape的意思，view之后就是[batch_size，kernel_num]
        flatten = conved.view(batch_size,-1)
        flatten = self.dropout(flatten)
        output = self.fc(flatten)
        return output




if __name__ == "__main__":
    net = TextCNN()
    
    





        



