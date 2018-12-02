import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


#-------------------------- extracting data----------------------------#
def extract_data():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)
    return list(zip(x, y))


def load_evaluation_data():
    dset = np.loadtxt('./x_test.txt')
    labels = np.loadtxt('./y_test.txt')
    devset = []

    for x in dset:
        devset.append(np.reshape(x, (1, 784)))

    return list(zip(devset, labels))


def load_test_data():
    tset = np.loadtxt('./x4pred.txt')
    testset = []

    for x in tset:
        testset.append(x.reshape((1, 784)))

    return testset

# --------------------------------------distance functions--------------------------#
# hamming distance helper
def hamming_distance_between_vectors(v1, v2):
    """Count the # of differences between equal length strings str1 and str2"""
    sum = 0
    for ch1, ch2 in zip(v1, v2):
        sum += (1 - np.sign(ch1 * ch2))/2
    return sum


def get_closest_vector_with_hamming_distance(matrix, out_vector):
    rows = len(matrix)
    # setting min to number of cols
    min_distance = len(matrix[0])
    min_index = 0
    i = 0
    # iterate over rows of matrix
    for r in range(rows):
        curr_distance = hamming_distance_between_vectors(matrix[r], out_vector)
        if curr_distance < min_distance:
            min_index = i
            min_distance = curr_distance
        i = i + 1
    return min_index


# loss based distance
def get_closest_vector_with_loss_based(matrix, out_vector):
    rows = len(matrix)
    # setting min to number of cols
    min_distance = len(matrix[0])
    #iterate over rows of matrix
    min_index = 0
    i = 0
    for r in range(rows):
        curr_distance = loss_based_between_vectors(matrix[r], out_vector)
        if curr_distance < min_distance:
            min_distance = curr_distance
            min_index = i
        i = i + 1
    return min_index


# loss based helper
def loss_based_between_vectors(v1, v2):
    total = 0
    for c1, c2 in zip(v1, v2):
        curr = 1 - (c1 * c2)
        curr = np.max([0, curr])
        total += curr
    return total

#--------------------------------------------predictions-------------------------#
def create_pred_vector(models, x, string):
    output_vector = [] # hamming distance
    pred_vector = [] # loss distance
    for model in models:
        y_tag, pred = model.get_pred_and_sign(x)
        pred_vector.append(pred)
        output_vector.append(y_tag)

    if "hamming" in string:
        return output_vector
    else:
        return pred_vector


# evaluating the dev set
def evaluate_dev(data, ecoc_mat, models, loss_function, string):
    y_pred =[]
    correct = 0
    incorrect = 0
    for x, y in data:
        prediction_vector = create_pred_vector(models, x, string)
        y_tag = loss_function(ecoc_mat, prediction_vector)
        if y_tag == y:
            correct = correct + 1
        else:
            incorrect = incorrect + 1

    accuracy = (correct * 100 * 1.0)/(len(data) * 1.0)

    print("Accuracy of " + string + ": " + str(accuracy))


def predict_results(dataa, ecoc_mat, models,filePath, string, loss_function):
    y_list = []

    for x in dataa:
        prediction_vector = create_pred_vector(models, x, string)
        y_tag = loss_function(ecoc_mat, prediction_vector)
        y_list.append(str(y_tag))
    # save results

    with open(filePath, 'w+') as f:
        f.write("\n".join(y_list))


# -----svm section----- #
class SVM:

    def __init__(self, lamda, weight, lrn, epoch):
        self.lamda = lamda
        self.W = weight
        self.lrn = lrn
        self.ep = epoch
        self.algorithms = {0:Algorithm('Loss', get_closest_vector_with_loss_based),
                           1:Algorithm('Hamming', get_closest_vector_with_hamming_distance)}

    def train(self, train_data):
        #lrn = self.lrn * (1 / self.ep)
        #loss = 0
        for t in range(1, self.ep + 1):
            loss = 0
            np.random.shuffle(train_data)
            lrn = self.lrn / np.sqrt(t)
            for x, y in train_data:
                wx = np.dot(x, self.W)
                pred = y * wx
                reg = lrn * self.lrn * self.W
                if 1 >= pred:
                    self.W += np.reshape(lrn * y * x, (784, 1)) - reg
                else:
                    self.W -= reg
                loss += pred[0]
            #print loss

    def get_pred_and_sign(self, x):
        pred = np.dot(x, self.W)
        y_sign = np.sign(pred)
        return y_sign, pred


class Algorithm:
    def __init__(self, name, method):
        self.name = name
        self.method = method
