# 收集数据：提供文本文件。
# 准备数据：编写函数 img2vector(), 将图像格式转换为分类器使用的向量格式
# 分析数据：在 Python 命令提示符中检查数据，确保它符合要求
# 训练算法：此步骤不适用于 KNN
# 测试算法：编写函数使用提供的部分数据集作为测试样本，测试样本与非测试样本的
#          区别在于测试样本是已经完成分类的数据，如果预测分类与实际类别不同，
#          则标记为一个错误
# 使用算法：本例没有完成此步骤，若你感兴趣可以构建完整的应用程序，从图像中提取
#          数字，并完成数字识别，美国的邮件分拣系统就是一个实际运行的类似系统

import os
import numpy as np
import heapq
import collections
import matplotlib.pyplot as plt


def read_data(path1):
    path = os.walk(path1)
    pathList = []
    LabelList = []
    dataList = []
    for _, _, x in path:
        for y in x:
            pathList.append("data/trainingDigits/" + y)
            LabelList.append(int(y[0]))

    for txtPath in pathList:
        with open(txtPath, "r", encoding="utf-8") as f:
            L = np.array([[int(y) for y in list(x.strip())] for x in f])
            L = np.ravel(L)
            dataList.append(L)
    return np.array(dataList), np.array(LabelList)




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
    trainDataList, trainLabelList = read_data("data/trainingDigits")
    trainDataList, trainLabelList = read_data("data/testDigits")
    # result = test(trainDataList, trainLabelList,trainDataList, trainLabelList, 5 )
    # print(result)
    result = np.array([[i, test(trainDataList, trainLabelList,trainDataList, trainLabelList, i)] for i in range(1, 15, 2)])
    print(result)
    # plt.scatter(result[:,0],result[:, 1])
    plt.plot(result[:, 0], result[:, 1], 'ro-')
    plt.xticks(np.arange(1, 15, 2))
    plt.show()


