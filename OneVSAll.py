import numpy as np
import itertools as it
import Ex1 as svm


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
for i in range(num_classes):
        new_data = svm.change_tags_ova(i, data)
        W = np.zeros((dim, 1))
        model = svm.SVM(lamdaa, W, lrn, epoch)
        model.train(new_data)
        model_list.append(model)
        print("finished with #{} model".format(i))

# evaluation of dev
dev_data = svm.load_evaluation_data()
svm.evaluate_dev(dev_data, ecoc_matrix, model_list, svm.get_closest_vector_with_hamming_distance, "ova with hamming")
svm.evaluate_dev(dev_data, ecoc_matrix, model_list, svm.get_closest_vector_with_loss_based, "ova with loss")

# predictions
test_data = svm.load_test_data()
svm.predict_results(test_data, ecoc_matrix, model_list, 'test.onevall.ham.pred', "hamming", svm.get_closest_vector_with_hamming_distance) # write hamming func
svm.predict_results(test_data, ecoc_matrix, model_list, 'test.onevall.loss.pred', "loss", svm.get_closest_vector_with_loss_based) # write loss func
