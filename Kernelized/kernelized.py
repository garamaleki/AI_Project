import numpy as np
from numpy import linalg

from utils import mnist_reader

train_loop = []
test_loop = []
train_h = []
train_v = []
test_h = []
test_v = []
with open('../extra/extra_train_loop_3.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            train_loop.append(int(x))

with open('../extra/extra_train_loop_3.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            train_loop.append(int(x))

with open('../extra/extra_train_h.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            train_h.append(np.array(list(map(int, x.split(',')))))
with open('../extra/extra_train_v.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            train_v.append(np.array(list(map(int, x.split(',')))))

with open('../extra/extra_test_loop_3.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            test_loop.append(int(x))

with open('../extra/extra_test_h.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            test_h.append(np.array(list(map(int, x.split(',')))))
with open('../extra/extra_test_v.txt', 'r') as file:
    for r in file:
        x = r.rstrip()
        if len(x) > 0:
            test_v.append(np.array(list(map(int, x.split(',')))))


def pixels_per_width(array_index, train=True):
    global train_v, test_v
    if train:
        return train_v[array_index]
    return test_v[array_index]


def pixels_per_height(array_index, train=True):
    global train_h, test_h
    if train:
        return train_h[array_index]
    return test_h[array_index]


def num_of_loops(array_index, train=True):
    global train_loop, test_loop
    if train:
        return train_loop[array_index]
    return test_loop[array_index]


def feature_extract(img_array, i, train=True):
    h = pixels_per_height(array_index=i, train=train)
    v = pixels_per_width(array_index=i, train=train)
    img_array = np.concatenate((h / 12, img_array))
    img_array = np.concatenate((v / 12, img_array))
    loops = num_of_loops(array_index=i, train=train)
    if loops == 0:
        return np.concatenate(([0, 0, 0, 0, 0], img_array))
    elif loops == 1:
        return np.concatenate(([1, 0, 0, 0, 0], img_array))
    elif loops == 2:
        return np.concatenate(([0, 1, 0, 0, 0], img_array))
    elif loops == 3:
        return np.concatenate(([0, 0, 1, 0, 0], img_array))
    elif loops == 4:
        return np.concatenate(([0, 0, 0, 1, 0], img_array))
    else:
        return np.concatenate(([0, 0, 0, 0, 1], img_array))


class Perceptron:

    def __init__(self, labels, kernel, sample_size, test_size, X, T):
        self.X = X
        self.sample_size = sample_size
        self.kernel = kernel
        self.labels = labels
        self.a = [[0 for i in range(sample_size)] for j in range(len(labels))]
        self.biases = [0 for i in range(len(labels))]
        self.K = np.zeros((sample_size, sample_size))
        self.T = np.zeros((sample_size, test_size))
        for i in range(sample_size):
            for j in range(sample_size):
                self.K[i, j] = self.kernel(feature_extract(self.X[i] / 255, i, train=True),
                                           feature_extract(self.X[j] / 255, j, train=True))
            if i % 10 == 0:
                print("kernel calc train: ", i)
        for i in range(sample_size):
            for j in range(test_size):
                self.T[i, j] = self.kernel(feature_extract(T[j] / 255, j, train=False),
                                           feature_extract(self.X[i] / 255, i, train=True))
            if i % 10 == 0:
                print("kernel calc test: ", i)

    def fit(self, size):
        pass

    def train(self, input, label, sample_index):
        predicted_label = self.predict(input, sample_index)
        if predicted_label == label:
            return 1
        self.a[predicted_label][sample_index] -= 1
        self.a[label][sample_index] += 1
        self.biases[predicted_label] -= 1
        self.biases[label] += 1
        return 0

    def predict(self, input, sample_index):
        scores = []
        for i in range(len(self.labels)):
            score = np.sum(self.K[:, sample_index] * self.a[i])
            scores.append(score + self.biases[i])
        index = np.argmax(scores)
        return self.labels[index]

    def predict_test(self, input, test_index):
        print(test_index)
        scores = []
        for i in range(len(self.labels)):
            score = np.sum(self.T[:, test_index] * self.a[i])
            scores.append(score + self.biases[i])
        index = np.argmax(scores)
        return self.labels[index]


def linear_kernel(x1, x2):
    return np.dot(x1, x2)


def polynomial_kernel(x, y, p=2):
    return (1 + np.dot(x, y)) ** p


def gaussian_kernel(x, y, sigma=2.0):
    return np.exp(-linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))


X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

features = 28 * 28 + 5 + 28 + 28
sample_size = 1000
test_size = 200
perceptron = Perceptron([i for i in range(10)], gaussian_kernel, sample_size, test_size, X_train, X_test)
epochs = 30

for j in range(epochs):
    correct = 0
    for i in range(sample_size):
        f = feature_extract(X_train[i] / 255, i, train=True)
        correct += perceptron.train(f, y_train[i], i)
        if i % 10 == 0:
            print("train: ", i)
    print(j, correct / sample_size)

correct = 0
for i in range(test_size):
    f = feature_extract(X_test[i] / 255, i, train=False)
    if perceptron.predict_test(f, i) == y_test[i]:
        correct += 1
print("sdf", correct / test_size)