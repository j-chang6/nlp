#读取自己的训练样本，进行padding 和 numerical
#设置max_len = 20,<unk> 用0 填充，<pad> 用1来填充


import random
import numpy as np
import os
from config import *


#获得标签值对应的索引
def getLabel2ind():
    label_w2n = {}
    label_n2w = {}
    with open(TNNConfig.labelFile,"r",encoding ="utf-8") as fp:
        line = fp.readline()
        while line :
            line = line.split()
            label_w2n[line[0]] = int(line[1])
            label_n2w[int(line[1])] = line[0]
            line = fp.readline()
    return label_w2n,label_n2w

#建立预训练词到词向量的映射
def getWord2Vec():
    word2vec = {}
    with open(TNNConfig.w2v_file,"r",encoding = "utf-8") as fp:
        line = fp.readline()
        line = fp.readline()
        while line:
            line  = line.split()
            word = str(line[0])
            embedding = np.asarray(line[1:],dtype = 'float32')
            word2vec[word] = embedding
            line = fp.readline()
    return word2vec


#建立词到索引的映射
def getWord2ind():
    wordw2n = {}
    wordw2n["<pad>"] = [0]*400
    wordw2n["<blank>"] = [0.01]*400
    with open(TNNConfig.wordList,'r',encoding = "utf-8") as f:
        line = f.readline().split()
        while line:
            wordw2n[line[0]] = int(line[1])
            line = f.readline().split()
    wordn2w = {wordw2n[w]:w for w in wordw2n}
    return wordw2n,wordn2w



#获得词表
def getWordList():
    wordvoc = []
    with open(TNNConfig.myTrainFile,"r",encoding = "utf-8") as fp:
        line = fp.readline().split()[1:]
        while line:
            wordvoc.extend(line)
            line = fp.readline().split()[1:]
        
    wordvoc = sorted(list(set(wordvoc)))
    wordvoc.insert(0,"<pad>")
    wordvoc.insert(1,"<blank>")
    with open(TNNConfig.wordList,"w",encoding = "utf-8") as f:
        for i in range(len(wordvoc)):
            f.write(wordvoc[i]+" {}\n".format(i))


#储存预训练模型的word_embedding_matrix矩阵
#首先读取wordList文件，然后根据词找到预训练的词向量，word_embedding_matrix的索引i出赋值该词向量

def load_embedding(w2v_npy,word_dim):
    wordw2n,wordn2w = getWord2ind()
    word2vec = getWord2Vec()

    word_embedding_matrix = np.zeros(shape = [len(wordw2n),word_dim],dtype = "float32")
    for w,i in wordw2n.items():
        if w in word2vec:
            word_embedding  = word2vec[w]
            word_embedding_matrix[i] = word_embedding
        else:
            word_embedding_matrix[i] = np.asarray([0] * word_dim ,dtype = "float32")
    
    #保存词向量
    if not os.path.exists(w2v_npy):
        embed_array = np.array(word_embedding_matrix,dtype = "float32")
        np.save(w2v_npy,embed_array)


#把训练数据的文本变成索引
def txt2ind(file):
    wordw2n,wordn2w = getWord2ind()
    labelw2n,labeln2w = getLabel2ind()
    maxLen = TNNConfig.seq_len

    x_input =[]
    with open(file,"r",encoding= "utf-8") as f:
        line = f.readline().split()
        while line:
            t = []
            t.append(labelw2n[line[0]])

            for w in line[1:]:
                t.append(wordw2n[w])
            
            #进行padding
            if len(t) > maxLen+1:
                t = t[:maxLen+1]
            elif len(t) < maxLen:
                t.extend([0]*(maxLen-len(t)+1))
            x_input.append(t)
            line = f.readline().split()
    return x_input




        


if __name__ == "__main__":
    # getWordList()

    # load_embedding(TNNConfig.pre_w2v_npy,50)
    # res = np.load(TNNConfig.pre_w2v_npy)
    # print(res.shape)
    
    x_input = txt2ind(TNNConfig.myTrainFile)
    for x in x_input:
        print(*x)

    
