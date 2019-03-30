# KNN
# 收集数据：提供文本文件
# 准备数据：使用 Python 解析文本文件
# 分析数据：使用 Matplotlib 画二维散点图
# 训练算法：此步骤不适用于 k-近邻算法
# 测试算法：使用海伦提供的部分数据作为测试样本。
#         测试样本和非测试样本的区别在于：
#             测试样本是已经完成分类的数据，如果预测分类与实际类别不同，则标记为一个错误。
# 使用算法：产生简单的命令行程序，然后海伦可以输入一些特征数据以判断对方是否为自己喜欢的类型。

# 读取数据，份训练数据和测试数据

# show data
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import heapq
import collections

datas = []
labels = []
with open("data/datingTestSet2.txt") as f:
    for sentence in f:
        L = sentence.strip().split("\t")
        datas.append([float(x) for x in L[:-1]])
        labels.append(int(L[-1]))
datas = np.array(datas)
labels = np.array(labels)
# show
# plt.scatter(datas[:,0], datas[:,2],20.0*labels, 20.0*labels)
# plt.show()

# 归一化
Max = np.array([np.max(datas[:,i]) for i in range(datas.shape[1])])
Min = np.array([np.min(datas[:,i]) for i in range(datas.shape[1])])
normalizeData = (datas - Min)/(Max-Min)

def distance(pos1, pos2):
    return np.sqrt(np.sum(np.square(pos1 - pos2), axis = 0))


def classify(trainData, label,  testData, k):
    D = [{'label':label[i] ,'distance': distance(trainData[i], testData)} for i in range(trainData.shape[0])]
    need = heapq.nsmallest(k, D, key= lambda s : s['distance'])
    realResult = [x["label"] for x in need]
    result = collections.Counter(realResult)
    return (result.most_common()[0][0])

def test(trainData, label,  testDataList,testLabelList, k):
    count = 0
    for id , testData in enumerate(testDataList):
        testData = np.array(testData)
        getLabel = classify(trainData, label, testData, k)
        if getLabel == testLabelList[id]:
            count +=1

    return (count/ (testLabelList.shape[0]+0.0))

if __name__ == '__main__':
    result = np.array([[i ,test(normalizeData[:800,], labels[:800 , ] , normalizeData[800:],labels[800: , ] , i) ] for i in range(1,15,2)])
    print(result)
    # plt.scatter(result[:,0],result[:, 1])
    plt.plot(result[:, 0], result[:, 1],'ro-')
    plt.xticks(np.arange(1,15,2))
    plt.show()



