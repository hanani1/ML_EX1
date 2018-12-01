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


# oa creation helper
def create_oa_matrix(size):
    res = []
    for x in range(size):
        curr = [-1] * size
        curr[x] = 1
        res.append(curr)
    return res

# return the combination of classes
def get_list_of_pairs_combi(i):
    oppsiteClasses = [j for j in range(4) if j != i]
    return it.combinations(oppsiteClasses,2)


# -----main----- #
num_classes = 4
lamdaa = 0.1
lrn = 0.05
dim = 784
epoch = 10
model_list = []

# -----One vs All----- #
data = svm.extract_data()
ecoc_matrix = create_oa_matrix(num_classes)

# -----training section----- #
for x in range(num_classes):
        new_data = svm.change_tags(x, data)
        W = np.zeros((dim, 1))
        model = svm.SVM(lamdaa, W, lrn, epoch)
        model.train(new_data)
        model_list.append(model)
        print("finished with #{} model".format(x))
oa_matrix = create_oa_matrix(num_classes)


# -----testing section----- #
test_data = load_test_data()
svm.SVM.evaluation(test_data, ecoc_matrix, model_list, 1, 'test.onevall.ham.pred', "hamming") # write hamming func
svm.SVM.evaluation(test_data, ecoc_matrix, model_list, 0, 'test.onevall.loss.pred', "loss") # write loss func
