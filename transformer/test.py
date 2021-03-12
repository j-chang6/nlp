#-*- coding = utf-8 -*-
import torch
import torch.nn
from loadData import *
import os
from config import Config
import matplotlib as plt
from model import *

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()

def translate_sentence(sentence,  model, device, src_i2w,tgt_i2w,max_len = 50):
    
    model.eval()
        

    tokens = ["<bos>"] + tokens + ["<eos>"]
    
    src_w2i = {w:i for i,w in src_i2w.items()}
    src_indexes = [src_w2i[token] for token in tokens]
    #[B,L]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    #<bos> index
    trg_indexes = [2]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        #<eos> index
        if pred_token == 4:
            break
    
    trg_tokens = [tgt_i2w[i] for i in trg_indexes]
    
    return trg_tokens[1:], attention



def main():
    src = ['a', 'woman', 'with', 'a', 'large', 'purse', 'is', 'walking', 'by', 'a', 'gate', '.']
    prep  = Preprocess() 
    src_train_sents, src_test_sents, src_val_sents, \
    tgt_train_sents, tgt_test_sents, tgt_val_sents = prep.forward(Config.src_train,Config.tgt_train,  
                                                             Config.src_test, Config.tgt_test,
                                                             Config.src_val, Config.tgt_val,
                                                             Config.src_vocab_file, Config.tgt_vocab_file)
    
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
    print("init model....")
    if os.path.exists(Config.weightFile):
        print("load model...")
        model.load_state_dict(torch.load(Config.weightFile))
    else:
        print("no model file")
        exit()
    

    translation, attention = translate_sentence(src,model, device,prep.src_i2w,prep.tgt_i2w)

    print(f'predicted trg = {translation}')
    display_attention(src, translation, attention)




if __name__ =="__main__":
    main()

    """
    ['eine', 'frau', 'mit', 'einer', 'großen', 'geldbörse', 'geht', 'an', 'einem', 'tor', 'vorbei', '.']
    """