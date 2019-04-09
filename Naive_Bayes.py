import numpy as np
import general as g
import loadData
import plot

# convert the Iris class to float
class Naive_Bayes:
    def __init__(self, attrNum, classNum):
        self.attrNum = attrNum
        self.classNum = classNum
        self.prior = np.zeros([1, classNum])


model = Naive_Bayes(32, 2)

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

def summarizeByClass(dataset):
    separated = separateByClass(dataset)

    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = g.summarize(instances)
    # print(summaries)
    return summaries

def predict(summaries, inputVector):
    probabilities = g.calculateClassProbabilities(summaries, inputVector)

    prob_parent = 0
    for classValue, probability in probabilities.items():
        # print(probabilities[int(classValue)] * model.prior[0][int(classValue-1)])
        prob_parent += probabilities[int(classValue)] * model.prior[0][int(classValue-1)]
        probabilities[classValue] = probabilities[int(classValue)] * model.prior[0][int(classValue-1)]

    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            probabilities[classValue] /= prob_parent
            bestProb = probability
            bestLabel = classValue
    # print(probabilities)
    return bestLabel, probabilities[2]


def getPredictions(summaries, testSet):
    predictions = []
    prob = []
    for i in range(len(testSet)):
        result, tmp = predict(summaries, testSet[i])
        predictions.append(result)
        prob.append(tmp)
    # print(predictions)
    return predictions, prob


def main():
    splitRatio = 0.67
    # dataset = loadData.loadWine()
    dataset = loadData.loadIono()
    trainingSet, testSet = g.splitDataset(dataset, splitRatio)

    summaries = summarizeByClass(trainingSet)
    # print(summaries)
    predictions,result_prob = getPredictions(summaries, testSet)
    x, y = g.splitXandY(testSet, model.attrNum, len(testSet))
    confusion_dim = len(summaries)
    confusion_matrix = np.zeros((confusion_dim, confusion_dim))
    accuracy, confusion_matrix = g.getAccuracy(testSet, predictions, confusion_matrix)
    print(accuracy)

    plot.ROC(y, result_prob)

main()



