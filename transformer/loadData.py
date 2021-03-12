#加载数据

from config import Config
from torch.utils.data import Dataset,DataLoader
from config import Config


class Preprocess():
    def __init__(self):
        self.src_w2i = dict()
        self.tgt_w2i = dict()
        self.src_i2w = dict()
        self.tgt_i2w = dict()
        self.src_max_len = 0
        self.tgt_max_len = 0
    

    #读取数据�?
    def read_file(self,filename):
        sents = []
        with open(filename,"r",encoding = "utf-8") as f:
            for sent in f.readlines():
                sents.append(['<bos>']+sent.strip().split()+['<eos>'])
        return sents
    
    #读取词表
    def read_voca_i2w(self,filename):
        voca = {0:'<unk>', 1:'<pad>', 2:'<bos>', 4:'<eos>'}
        i = 5
        with open(filename,"r",encoding = "utf-8") as f:
            for line in f.readlines():
                voca[i] = line.split()[0]
                i+=1
        return voca
    
    #获取最大长�?
    def get_max_len(self,sents):
        #这里的sent是一个list
        max_len = max([len(sent) for sent in sents])
        return max_len

    #索引�?�?padding
    def txt_idx(self,sents,label):
        idxs = []
        if label == "src":
            for sent in sents:
                t = [self.src_w2i[word] if word in self.src_w2i else 0 for word in sent ]
                t = self.add_pad(t,self.src_max_len)
                idxs.append(t)
        else:
            for sent in sents:
                t= [self.tgt_w2i[word] if word in self.tgt_w2i else 0 for word in sent ]
                t = self.add_pad(t,self.tgt_max_len)
                idxs.append(t)
        return idxs

    #进行padding
    def  add_pad(self,sent,max_len):
        if len(sent) > max_len:
            return sent[:max_len]
        else:
            return sent + [1] * (max_len-len(sent))

    
    
    #进行预处�?
    def forward(self,src_train,tgt_train,
                src_test,tgt_test,
                src_val,tgt_val,
                src_voca_file,tgt_voca_file):
        src_train_sents = self.read_file(src_train)
        tgt_train_sents = self.read_file(tgt_train)
        src_test_sents = self.read_file(src_test)
        tgt_test_sents = self.read_file(tgt_test)
        src_val_sents = self.read_file(src_val)
        tgt_val_sents = self.read_file(tgt_val)

        #获得词表
        self.src_i2w = self.read_voca_i2w(src_voca_file)
        self.tgt_i2w = self.read_voca_i2w(tgt_voca_file)
        
        #获得词表w2i
        self.src_w2i = {w:i for i,w in self.src_i2w.items()}
        self.tgt_w2i = {w:i for i,w in self.tgt_i2w.items()}


        self.src_max_len = self.get_max_len(src_train_sents)
        self.tgt_max_len = self.get_max_len(tgt_train_sents)

        #进行numerical
        src_train_sents, src_test_sents, src_val_sents = self.txt_idx(src_train_sents, 'src'), \
                                                         self.txt_idx(src_test_sents, 'src'), \
                                                         self.txt_idx(src_val_sents, 'src')
        tgt_train_sents, tgt_test_sents, tgt_val_sents = self.txt_idx(tgt_train_sents, 'tgt'), \
                                                         self.txt_idx(tgt_test_sents, 'tgt'), \
                                                         self.txt_idx(tgt_val_sents, 'tgt')
        return src_train_sents,src_test_sents,src_val_sents,\
               tgt_train_sents,tgt_test_sents,tgt_val_sents


class MyDataset(Dataset):
    def __init__(self,src,tgt):
        self.src = src
        self.tgt = tgt
    
    def __getitem__(self,index):
        return self.src[index] ,self.tgt[index]

    def __len__(self):
        return len(self.src)

def collate_fn(batch_data):
    src,tgt  = list(zip(*batch_data))
    return torch.LongTensor(src) ,torch.LongTensor(tgt)

def build_vocab(dr_in,dr_out):
    dic = {}
    with open(dr_in,"r",encoding = "utf-8") as f:
        for sent in f.read().split("\n"):
            for token in sent:
                if token in dic:
                    dic[token]+=1
                else:
                    dic[token] = 1
    
    #sorted
    lst = sorted(dic.items(),key = lambda x:-x[1])

    with open(dr_out,"w",encoding="utf-8") as f:
        for token,num in dic.items():
            f.write("{} {}\n".format(token,num))

    
    #sorted
    


if __name__ =="__main__":
    # prep = Preprocess()
    # src_train_sents, src_test_sents, src_val_sents, \
    # tgt_train_sents, tgt_test_sents, tgt_val_sents = prep.forward(Config.src_train,Config.tgt_train,  
    #                                                          Config.src_test, Config.tgt_test,
    #                                                          Config.src_val, Config.tgt_val,
    #                                                          Config.src_vocab_file, Config.tgt_vocab_file)
    