class Config():
    src_train = r'data/test.en.40000'
    tgt_train = r'data/train.de.40000'
    src_test = r'data/test.en.40000'
    tgt_test = r'data/test.de.40000'
    src_val = r'data/val.en.40000'
    tgt_val = r'data/val.de.40000'
    src_vocab_file = r'data/vocab.en.40000'
    tgt_vocab_file = r'data/vocab.de.40000'

    weightFile = r"model.pth"
    predict_file = r"data/pre"
    out_file = r"data/out"

    lr = 0.0005
    gamma = 0.05
    epoch = 60
    num_beams =  8
    pad_index = 1
    d_model = 512
    d_ff = 2048
    dropout = 0.1
    d_k = d_v = 64
    n_layers = 6
    n_heads = 8
    CLIP = 1