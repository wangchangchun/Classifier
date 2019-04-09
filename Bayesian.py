import numpy as np
import general as g
import loadData
import plot
import random
import math

class Bayesian:
    def __init__(self, attrNum, classNum):
        self.attrNum = attrNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])


model = Bayesian(32, 2)


def meanVector(dataset):
    mean = [g.mean(attribute) for attribute in zip(*dataset)]
    # del summaries[-1]
    # print(mean)
    return mean


def stdMat(dataset):
    std = np.zeros((len(meanVector(dataset)), len(meanVector(dataset))))
    miu = np.array(meanVector(dataset))
    for i in range(len(dataset)):
        diff = (dataset[i] - miu)[:, None]
        std += diff * diff.T
    std /= len(dataset)
    std += np.eye(model.attrNum) * 1e-15
    print(np.shape(std))
    return std


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    # print(separated)
    model.prior = g.prior(separated,model.classNum)
    return separated


# separate the data by class
# calculate the each class's mean and standard variance of each attribute (column)
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        x, y = g.splitXandY(instances, model.attrNum, len(instances))
        summaries[classValue] = meanVector(x), stdMat(x)
    return summaries




def multi_gaussian_pdf(x, mean, covar):
    from math import exp, sqrt, log, pi
    from numpy.linalg import inv, det
    n = np.array(mean).shape[0]
    test_data = np.zeros((1, n))
    for i in range(n):
        test_data[0][i] = x[i]
    exp_term = -(1/2) * (test_data-mean).dot(inv(covar)).dot((test_data-mean).T)
    const_term = 1/((2*pi)**(n/2)*(det(covar)**(1/2)))
    # print(const_term * np.asscalar(np.exp(exp_term)) )
    return const_term * np.asscalar(np.exp(exp_term))




def getPredictions(summaries, testSet):
    predictions = []
    result_prob = np.zeros((len(testSet), model.classNum))
    for i in range(len(testSet)):
        parent = 0

        max_prob_class = 0
        best_prob = 0
        for j in range(model.classNum):
            parent += multi_gaussian_pdf(testSet[i], summaries[j + 1][0], summaries[j + 1][1]) \
                      * model.prior[0][j]
        for j in range(model.classNum):
            result_prob[i][j] = (multi_gaussian_pdf(testSet[i], summaries[j + 1][0], summaries[j + 1][1])
                                * model.prior[0][j] / parent)
            if result_prob[i][j] > best_prob :
                best_prob = result_prob[i][j]
                max_prob_class = j

        predictions.append(max_prob_class+1)
    print(predictions)
    return predictions , result_prob





def main():
    splitRatio = 0.67
    dataset = loadData.loadIono()
    trainingSet, testSet = g.splitDataset(dataset, splitRatio)

    summaries = summarizeByClass(trainingSet)

    predictions , result_prob = getPredictions(summaries, testSet)

    x, y = g.splitXandY(testSet, model.attrNum, len(testSet))
    # print(y)

    confusion_matrix = np.zeros((len(summaries), len(summaries)))
    accuracy, confusion_matrix= g.getAccuracy(testSet, predictions, confusion_matrix)
    print(accuracy)

    plot.ROC(y, result_prob[:, 1])


main()




