import torch
import torch.nn as nn
from loadData import *
from torch.optim.lr_scheduler import ExponentialLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,Dataset
from torch import optim
import os
from config import Config
from model import *
import pdb
import time
import math


def train(model, iterator, optimizer, criterion, scheduler):
     
    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(dataLoader):
        
        src = batch[0]
        trg = batch[1]
        
        optimizer.zero_grad()
        
        output, _ = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch[0]
            trg = batch[1]

            output, _ = model(src, trg[:,:-1])
            
            #output = [batch size, trg len - 1, output dim]
            #trg = [batch size, trg len]
            
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            
            #output = [batch size * trg len - 1, output dim]
            #trg = [batch size * trg len - 1]
            
            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def main():
    #预处理
    prep  = Preprocess() 
    src_train_sents, src_test_sents, src_val_sents, \
    tgt_train_sents, tgt_test_sents, tgt_val_sents = prep.forward(Config.src_train,Config.tgt_train,  
                                                             Config.src_test, Config.tgt_test,
                                                             Config.src_val, Config.tgt_val,
                                                             Config.src_vocab_file, Config.tgt_vocab_file)

    trainset = MyDataset(src_train_sents,tgt_train_sents)
    train_dataLoader = DataLoader(trainset,batch_size = 64,shuffle = True,collate_fn = collate_fn)
    valset = MyDataset(src_val_sents,tgt_val_sents)
    val_dataLoader = DataLoader(valset,batch_size=64,shuffle=True,collate_fn=collate_fn)

    #创建模型
    best_valid_loss = float('inf')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    enc = Encoder(
                len(prep.src_i2w),
                Config.d_model,
                Config.n_layers,
                Config.n_heads,
                Config.d_ff,
                Config.dropout,
                device)
    dec = Decoder(
                len(prep.tgt_i2w),
                Config.d_model,
                Config.n_layers,
                Config.n_heads,
                Config.d_ff,
                Config.dropout,
                device)
    model = Seq2Seq(enc,dec,Config.pad_index,device).to(device)


    #创建损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=1)
    optimizer = optim.Adam(model.parameters(),lr = Config.lr,betas=(0.9,0.99))
    scheduler = ExponentialLR(optimizer,Config.gamma**(1/Config.epoch))
    


    for epoch in range(Config.epoch):
        
        start_time = time.time()
        
        train_loss = train(model, train_dataLoader, optimizer, criterion, scheduler)
        valid_loss = evaluate(model, val_dataLoader, criterion)
        
        end_time = time.time()
        
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'best-model.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        scheduler.step()


if __name__ =="__main__":
    main()
