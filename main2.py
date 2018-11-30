import numpy as np
import itertools as it
import Ex1 as svm


def load_test_data():
    tset = np.loadtxt('x_test.txt')
    testset = []
    for x in tset:
        testset.append(np.reshape(x, (1, 784)))
    return testset


def create_oa_matrix(size):
    res = []
    for x in range(size):
        curr = [-1] * size
        curr[x] = 1
        res.append(curr)
    return res



def get_list_of_pairs_combi(i):
    oppsiteClasses = [j for j in range(4) if j != i]
    return it.combinations(oppsiteClasses, 2)

def update_ecoc_matrix(ecoc_matrix, classifier_index, y0, y1):
    ecoc_matrix[y0, classifier_index] = 1
    ecoc_matrix[y1, classifier_index] = -1

# main
num_classes = 4
lamdaa = 0.1
lrn = 0.05
dim = 784
epoch = 4

data = svm.extract_data()

# all pairs
num_of_models = int(num_classes * (num_classes-1) / 2)
all_pairs_matrix = np.zeros((num_classes, num_of_models), dtype=int)
model_list = []
pair_index = 0
for y0, y1 in (it.combinations(range(num_classes), 2)):
    update_ecoc_matrix(all_pairs_matrix, pair_index, y0, y1)
    pair_index = pair_index + 1
    new_data = svm.change_tags_all_pairs(y0, y1, data)
    W = np.zeros((dim, 1))
    model = svm.SVM(lamdaa, W, lrn, epoch)
    model.train(new_data)
    model_list.append(model)
# testing
test_data = load_test_data()
svm.SVM.evaluation(test_data, all_pairs_matrix, model_list, 1, 'test.onevall.ham.pred') # write hamming func
svm.SVM.evaluation(test_data, all_pairs_matrix, model_list, 0, 'test.onevall.loss.pred') # write loss func