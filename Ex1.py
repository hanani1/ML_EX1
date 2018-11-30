import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle


def extract_data():
    # read data
    mnist = fetch_mldata("MNIST original", data_home="./data")
    #eta = 0.1
    X, Y = mnist.data[:60000] / 255., mnist.target[:60000]
    x = [ex for ex, ey in zip(X, Y) if ey in [0, 1, 2, 3]]
    y = [ey for ey in Y if ey in [0, 1, 2, 3]]
    # suffle examples
    x, y = shuffle(x, y, random_state=1)
    return list(zip(x, y))

def change_tags_all_pairs(y0, y1, data):
    result = []
    for x, y in data:
        if(y == y0):
            result.append((x, 1))
        elif(y == y1):
            result.append((x, -1))
        else:
            result.append((x, 0))
    return result

def change_tags(index, data):
    result = []
    for x, y in data:
        if y == index:
            result.append((x, 1))
        else:
            result.append((x, -1))

    return result

# hamming distance
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


def loss_based_between_vectors(v1, v2):
    total = 0
    for c1, c2 in zip(v1, v2):
        curr = 1 - (c1 * c2)
        curr = np.max([0, curr])
        total += curr
    return total

# svm section
class SVM:

    def __init__(self, lamda, weight, lrn, epoch):
        self.lamda = lamda
        self.W = weight
        self.lrn = lrn
        self.ep = epoch
        self.algorithms = {0:Algorithem('Loss', get_closest_vector_with_loss_based),
                           1:Algorithem('Hamming', get_closest_vector_with_hamming_distance)}

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

    @staticmethod
    def evaluation(dataa, ecoc_mat, models, ok, filePath):
        y_list = []

        for x in dataa:
            vector = []
            i = 0
            for model in models:

                y_pred, pred = models[i].get_pred_and_sign(x)

                if models[i].algorithms[ok].name is "Loss":
                    vector.append(pred)
                else:
                    vector.append(y_pred)

                y_tag = models[i].algorithms[ok].method(ecoc_mat, vector)
                i += 1

                y_list.append(str(y_tag))

        # save results
        with open(''.join(filePath), 'w+') as f:
            f.write(" ".join(y_list))

    def get_pred_and_sign(self, x):
        pred = np.dot(x, self.W)
        y_sign = np.sign(pred)
        return y_sign, pred


class Algorithem:
    def __init__(self, name, method):
        self.name = name
        self.method = method
