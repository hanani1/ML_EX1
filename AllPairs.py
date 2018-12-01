import numpy as np
import itertools as it
import Ex1 as svm


# load data
def load_test_data():
    tset = np.loadtxt('x_test.txt')
    testset = []
    for x in tset:
        testset.append(np.reshape(x, (1, 784)))
    return testset


# create the combination with the classes
def get_list_of_pairs_combi(i):
    oppsiteClasses = [j for j in range(4) if j != i]
    return it.combinations(oppsiteClasses, 2)


def update_ecoc(ecoc_matrix, classifier, y0, y1):
    ecoc_matrix[y0, classifier] = 1
    ecoc_matrix[y1, classifier] = -1


# main #
num_classes = 4
lamdaa = 0.1
lrn = 0.05
dim = 784
epoch = 4

# --------- all pairs ---------- #
data = svm.extract_data()
num_of_models = int(num_classes * (num_classes-1) / 2)
all_pairs_matrix = np.zeros((num_classes, num_of_models), dtype=int)
model_list = []
pair = 0

# -- training section -- #
for y0, y1 in (it.combinations(range(num_classes), 2)):
    update_ecoc(all_pairs_matrix, pair, y0, y1)
    pair = pair + 1
    new_data = svm.change_tags_all_pairs(y0, y1, data)
    W = np.zeros((dim, 1))
    model = svm.SVM(lamdaa, W, lrn, epoch)
    model.train(new_data)
    model_list.append(model)

# -- testing section -- #
test_data = load_test_data()
svm.SVM.evaluation(test_data, all_pairs_matrix, model_list, 1, 'test.allpairs.ham.pred', "hamming")
svm.SVM.evaluation(test_data, all_pairs_matrix, model_list, 0, 'test.allpairs.loss.pred', "loss")