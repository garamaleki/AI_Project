import numpy as np

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
    img_array = np.concatenate((h * 25, img_array))
    img_array = np.concatenate((v * 25, img_array))
    loops = num_of_loops(array_index=i, train=train)
    if loops == 0:
        return np.concatenate(([0, 0, 0, 0, 0], img_array))
    elif loops == 1:
        return np.concatenate(([500, 0, 0, 0, 0], img_array))
    elif loops == 2:
        return np.concatenate(([0, 500, 0, 0, 0], img_array))
    elif loops == 3:
        return np.concatenate(([0, 0, 500, 0, 0], img_array))
    elif loops == 4:
        return np.concatenate(([0, 0, 0, 500, 0], img_array))
    else:
        return np.concatenate(([0, 0, 0, 0, 500], img_array))


class Perceptron:

    def __init__(self, labels, features):
        self.labels = labels
        self.w = [[0 for i in range(features)] for j in range(len(labels))]
        self.biases = [0 for i in range(len(labels))]

    def train(self, input, label):
        predicted_label = self.predict(input)
        if predicted_label == label:
            return 1
        self.w[predicted_label] = np.subtract(self.w[predicted_label], input)
        self.biases[predicted_label] -= 1
        self.w[label] = np.add(self.w[label], input)
        self.biases[label] += 1
        return 0

    def predict(self, input):
        index = np.argmax(
            [(np.dot(self.w[i], input) + self.biases[i]) for i in range(len(self.labels))])
        return self.labels[index]


X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')

features = 28 * 28 + 5 + 28 + 28
perceptron = Perceptron([i for i in range(10)], features)
epochs = 30

for j in range(epochs):
    correct = 0
    for i in range(len(X_train)):
        f = feature_extract(X_train[i], i, train=True)
        correct += perceptron.train(f, y_train[i])
    print(j, correct / len(X_train))

x = 0
for i in range(len(X_test)):
    f = feature_extract(X_test[i], i, train=False)
    if perceptron.predict(f) == y_test[i]:
        x += 1

print(x / len(X_test))
