#对获得的中文语料库进行分词

import jieba.analyse
import codecs

ifp = codecs.open("zhwiki_jian_zh.txt","r",encoding = "utf-8")
ofp = codecs.open("wiki_jian_seg.txt","w",encoding = "utf-8")

print("start ....")
line = ifp.readline()
while line:
    words = jieba.cut(line)
    ofp.write(" ".join(words))
    ofp.write("\n")
    line = ifp.readline()

ifp.close()
ofp.close()