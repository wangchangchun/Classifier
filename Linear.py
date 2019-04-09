import numpy as np
import random
import general as g
import loadData
import plot

class Linear:

    def __init__(self, attrNum, classNum,batchNum):
        self.attrNum = attrNum
        self.batchNum = batchNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])

        self.lr = 0.5
        self.input_node = attrNum
        self.w = np.random.uniform(0,1, (attrNum, 1))


model = Linear(32, 2, 30)

def batch(dataset, batchNum):
    # print(batchNum)
    trainSize = int(batchNum)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return trainSet


def train(x, y):
    errNum = 0

    output = x.dot(model.w)
    answer = []
    for i in range(len(x)):

        if output[i][0] >= 0:
            answer .append(float(2))
            if y[i][0] == 1:
                errNum += 1

        else:
            answer .append(float(1))
            if y[i][0] == 2:
                errNum += 1
    # print(np.shape(answer - y.T))
    de_error = np.array(x.T .dot((answer - y.T).T))
    model.w = model.w - model.lr * de_error

    # print(np.shape(de_error))
    # print("error rate : ")
    # print(errNum / len(x))

    return output

def predict_test (x, y):
    y_prob =[]
    # print(np.shape(model.w))
    output = x.dot(model.w)
    max = output.max()
    min = output.min()
    answer = []
    correct = 0
    for i in range(len(y)):
        if output[i][0] >= 0:
            answer.append(float(2))
        else:
            answer.append(float(1))
        if answer[i] == y[i][0]:
            correct += 1
        prob = (output[i][0] - min) / (max - min)
        y_prob.append(prob)

    loss = np.mean((y - output) ** 2)

    # plot.plot_result(x, y,3,5, w=model.w)
    print("accuracy : ")
    print(correct/len(y)*100)
    confusion_matrix = np.zeros((model.classNum, model.classNum))
    accuracy, confusion_matrix = g.getAccuracy(y, answer, confusion_matrix)

    return y_prob , accuracy

def main():
    splitRatio = 0.67
    dataset = loadData.loadIono()
    trainingSet, testSet = g.splitDataset(dataset, splitRatio)

    for i in range(5000):
        if i % 100 == 0:
            model.lr = model.lr/5
        batchData = batch(trainingSet, model.batchNum)
        x, y = g.splitXandY(batchData, model.attrNum, len(batchData))
        train(x, y)
    x, y = g.splitXandY(testSet, model.attrNum, len(testSet))
    final_output, accuracy = predict_test(x, y)
    plot.ROC(y, final_output)
    # plot.ROC_plot(y, final_output)
    # loss = np.mean((y - final_output) ** 2)
    return accuracy

main()


