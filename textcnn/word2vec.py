#使用gensim训练自己的词向量
import logging
import os.path
import sys
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

if __name__ =="__main__":
    #sys.argv[0]指的是路径加文件名，、
    #os.path.basename()是提取文件名
    program = os.path.basename(sys.argv[0])
    #给定一个名称，返回一个logger对象
    logger = logging.getLogger(program)
    #format指定输出的格式
    #%(asctime)s打印日志时间
    #%(levelname)s打印日志级别
    #(message)s打印日志的信息
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
    logging.root.setLevel(level = logging.INFO)

    #打印这是一个通知日志
    logger.info("runing %s" % " ".join(sys.argv))

    #检查程序输入参数
    if len(sys.argv) < 4:
        print(globals()['__doc__']%locals())
        sys.exit(1)
    
    #inp是分词好的文本，outp1是训练好的模型，outp2是训练好的词向量
    inp,outp1,outp2 = sys.argv[1:4]

    #linesentece(inp)表示一句话一行
    #size 是每个词的词向量
    #windows 词向量扫描窗口，为5则考验前5个和后5个
    #min_count设置最低频率的词
    #worker是进程数
    #sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW

    model = Word2Vec(LineSentence(inp),size= 400,window= 5,min_count = 5,workers = multiprocessing.cpu_count())
    model.save(outp1)
    #不以二进制存储
    model.wv.save_word2vec_format(outp2,binary =False)
