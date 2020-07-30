class TNNConfig():
    embedding = 50
    seq_len = 25
    filter_sizes = [3]

    num_class = 5
    vocab_size = 2166 #会自动修改
    num_fifters  =300
    dropout_keep = 0.5
    lr = 1e-3

    batch_size = 64
    num_epochs = 500


    trainFile = r"textcnn\data\trainquestion.txt"
    testFile = r"textcnn\data\testquestion.txt"
    myTrainFile = r"textcnn\data\my_trainquestion.txt"
    myTestFile = r"textcnn\data\my_testquestion.txt"
    wordList = r"textcnn\data\wordList.txt"
    labelFile = r"textcnn\data\label.txt"
    w2v_file = r"zhwiki_hlt_50d_word2vec_tmp10000"
    pre_w2v_npy = r"textcnn\data\w2v_npy.npy"

    weightFile = r"textcnn\data\weightFile"
