def ROC (y,result_prob):
    import sklearn.metrics as metrics
    FA, PD, threshold = metrics.roc_curve(y, result_prob, pos_label=2)
    roc_auc = metrics.auc(FA, PD)



    # print(FA, '\n', PD, '\n', threshold)
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')
    plt.plot(FA, PD, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
def plot_result(x, y,attr1 = 0, attr2 = 1, w=None):

    for i in range(len(x)):
        xs = x[i][attr1]
        ys = x[i][attr2]
        # plt.scatter(xs, ys)
        if(y[i][0]==1):
            plt.plot(xs, ys, 'ro')
        else:
            plt.plot(xs, ys, 'bo')

    xs = np.linspace(0, 1, 1000)
    ys = []
    w1, w2 = w[attr1][0], w[attr2][0]
    for x in xs:
        y = -(w1 * x) / w2
        ys.append(y)
        # print(y)
    plt.plot(xs, ys, 'r--')

    plt.xlim(0, 1), plt.ylim(0, 1)
    plt.show()
    #plt.legend(loc='best')

def ROC_plot(y, result_prob):
    import matplotlib.pyplot as plt
    plt.title('Receiver Operating Characteristic')

    AUC = 0
    old_FA_TD = [1, 1]
    T = 1e-50
    for i in range(50):
        T *= 10
        TP = 0
        TN = 0
        FP = 0
        FN = 0

        for j in range(len(result_prob)):
            if result_prob[j] > T:
                if y[j] == 2:
                    TP += 1
                else:
                    FP += 1
            else:
                if y[j] == 2:
                    FN += 1
                else:
                    TN += 1
        TD = TP / (TP + FN)
        FA = FP / (FP + TN)
        # print([FA, TD])
        AUC += (TD + old_FA_TD[1])*(old_FA_TD[0]-FA)/2
        old_FA_TD = [FA, TD]

        plt.plot(FA, TD, 'bo')
    print(AUC)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()








