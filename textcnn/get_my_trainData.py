#从语料库中获得自己的训练样本

import jieba 

train_dr = "textcnn/data/trainquestion.txt"
my_train_dr  ="textcnn/data/my_trainquestion.txt"
test_dr = "textcnn/data/testquestion.txt"
my_test_dr = "textcnn/data/my_testquestion.txt"

WantedNum = 200


def main():
    #读取textcnn\data\trainquestion.txt文件，然后保留标签和文本
    ifp = open(test_dr,"r",encoding = "utf-8")
    ofp = open(my_test_dr,"w",encoding= "utf-8")
    dic_label = {'DES': 0, 'HUM': 0, 'LOC': 0, 'NUM': 0, 'OBJ': 0}

    line = ifp.readline()
    num = 0
    while line:
        line = line.split()
        label = line[0][:3]
        words = " ".join(jieba.cut(line[1]))
        if label in dic_label and dic_label[label] < WantedNum:
            dic_label[label] +=1
            ofp.write(label+" "+words+"\n")
            num+=1

        if num >= WantedNum * 5:
            break
        line = ifp.readline()

        




if __name__ =="__main__":
    main()