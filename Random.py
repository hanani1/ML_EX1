import numpy as np
import itertools as it
import Ex1 as svm


def create_random_matrix():
    matrix = []
    first_row = {1, 0, 0, 0, 1, -1, 1, -1, 1}
    matrix.append(first_row)
    second_row = {0, 1, -1, 1, -1, -1, -1, 1, 1}
    matrix.append(second_row)
    third_row = {-1 ,-1, 1, 1, -1, -1, 1, -1, -1}
    matrix.append(third_row)
    fourth_row = {0, 1, 0, -1, -1, 1, -1, -1, 1}
    matrix.append(fourth_row)
    return matrix


def change_data_for_random_1(data):
    result = []
    for x, y in data:
        if y == 0:
            result.append((x, 1))
        elif y == 1:
            result.append((x, 0))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, 0))
    return result


def change_data_for_random_2(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 0))
        elif y == 1:
            result.append((x, 1))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, 1))
    return result


def change_data_for_random_3(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 0))
        elif y == 1:
            result.append((x, -1))
        elif y == 2:
            result.append((x, 1))
        else:
            result.append((x, 0))
    return result


def change_data_for_random_4(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 0))
        elif y == 1:
            result.append((x, 1))
        elif y == 2:
            result.append((x, 1))
        else:
            result.append((x, -1))
    return result


def change_data_for_random_5(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 1))
        elif y == 1:
            result.append((x, -1))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, -1))
    return result


def change_data_for_random_6(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, -1))
        elif y == 1:
            result.append((x, -1))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, 1))
    return result


def change_data_for_random_7(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 1))
        elif y == 1:
            result.append((x, -1))
        elif y == 2:
            result.append((x, 1))
        else:
            result.append((x, -1))
    return result


def change_data_for_random_8(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, -1))
        elif y == 1:
            result.append((x, 1))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, -1))
    return result


def change_data_for_random_9(data):
    result = []
    for x,y in data:
        if y == 0:
            result.append((x, 1))
        elif y == 1:
            result.append((x, 1))
        elif y == 2:
            result.append((x, -1))
        else:
            result.append((x, 1))
    return result

# -----main----- #
lamdaa = 0.1
lrn = 0.05
dim = 784
epoch = 20
model_list = []

data = svm.extract_data()
ecoc_matrix = create_random_matrix()
#----------training models---------------#
#first model
new_data = change_data_for_random_1(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#second model
new_data = change_data_for_random_2(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#third model
new_data = change_data_for_random_3(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#fourth model
new_data = change_data_for_random_4(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#5
new_data = change_data_for_random_5(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#6
new_data = change_data_for_random_6(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#7
new_data = change_data_for_random_7(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#8
new_data = change_data_for_random_8(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)
#9
new_data = change_data_for_random_9(data)
W = np.zeros((dim, 1))
model = svm.SVM(lamdaa, W, lrn, epoch)
model.train(new_data)
model_list.append(model)

# evaluation of dev
dev_data = svm.load_evaluation_data()
svm.evaluate_dev(dev_data, ecoc_matrix, model_list, svm.get_closest_vector_with_hamming_distance, "random with hamming")
svm.evaluate_dev(dev_data, ecoc_matrix, model_list, svm.get_closest_vector_with_loss_based, "random with loss")

# predictions
test_data = svm.load_test_data()
svm.predict_results(test_data, ecoc_matrix, model_list, 'test.random.ham.pred', "hamming", svm.get_closest_vector_with_hamming_distance)
svm.predict_results(test_data, ecoc_matrix, model_list, 'test.random.loss.pred', "loss", svm.get_closest_vector_with_loss_based)