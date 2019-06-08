import numpy as np
from skimage.measure import block_reduce

from utils import mnist_reader

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')


class Graph:
    # source: geeksforgeeks
    def __init__(self, row, col, g):
        self.ROW = row
        self.COL = col
        self.graph = g

    def isSafe(self, i, j, visited):
        return (i >= 0 and i < self.ROW and
                j >= 0 and j < self.COL and
                not visited[i][j] and not self.graph[i][j])

    def DFS(self, i, j, visited):

        rowNbr = [-1, -1, -1, 0, 0, 1, 1, 1]
        colNbr = [-1, 0, 1, -1, 1, -1, 0, 1]

        visited[i][j] = True

        for k in range(8):
            if self.isSafe(i + rowNbr[k], j + colNbr[k], visited):
                self.DFS(i + rowNbr[k], j + colNbr[k], visited)

    def countIslands(self):
        visited = [[False for j in range(self.COL)] for i in range(self.ROW)]

        count = 0
        for i in range(self.ROW):
            for j in range(self.COL):
                if visited[i][j] == False and self.graph[i][j] == 0:
                    self.DFS(i, j, visited)
                    count += 1

        return count


with open('extra_train_loop_3.txt', 'w') as file:
    for i in range(len(X_train)):
        img_2d = X_train[i].reshape((28, 28))
        x = block_reduce(img_2d, (3, 3), np.max)
        g2 = Graph(10, 10, x)
        c = g2.countIslands()
        print(i, c)
        file.write(str(c) + "\n")

with open('extra_test_loop_3.txt', 'w') as file:
    for i in range(len(X_test)):
        img_2d = X_test[i].reshape((28, 28))
        x = block_reduce(img_2d, (3, 3), np.max)
        g2 = Graph(10, 10, x)
        c = g2.countIslands()
        print(i, c)
        file.write(str(c) + "\n")

with open('extra_train_h.txt', 'w') as file:
    for k in range(len(X_train)):
        img_2d = X_train[k].reshape((28, 28))
        h = [0 for i in range(28)]
        for i in range(28):
            for j in range(28):
                if img_2d[i][j] > 0:
                    h[i] += 1
        print(k)
        file.write(','.join(list(map(str, h))) + "\n")

with open('extra_train_v.txt', 'w') as file:
    for k in range(len(X_train)):
        img_2d = X_train[k].reshape((28, 28))
        v = [0 for i in range(28)]
        for i in range(28):
            for j in range(28):
                if img_2d[i][j] > 0:
                    v[j] += 1
        print(k)
        file.write(','.join(list(map(str, v))) + "\n")

with open('extra_test_h.txt', 'w') as file:
    for k in range(len(X_test)):
        img_2d = X_test[k].reshape((28, 28))
        h = [0 for i in range(28)]
        for i in range(28):
            for j in range(28):
                if img_2d[i][j] > 0:
                    h[i] += 1
        print(k)
        file.write(','.join(list(map(str, h))) + "\n")

with open('extra_test_v.txt', 'w') as file:
    for k in range(len(X_test)):
        img_2d = X_test[k].reshape((28, 28))
        v = [0 for i in range(28)]
        for i in range(28):
            for j in range(28):
                if img_2d[i][j] > 0:
                    v[j] += 1
        print(k)
        file.write(','.join(list(map(str, v))) + "\n")
