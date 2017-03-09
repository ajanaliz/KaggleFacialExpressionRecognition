import numpy as np
from sklearn.utils import shuffle


def init_weight_and_bias(M1, M2): # M1 means input size and M2 means output size
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2) # we have a matrix of size M1xM2 which is randomized initially to a gaussian normal and we make the standard deviation of this the square root of M1+M2
    b = np.zeros(M2) # initialize the bias to zeros
    return W.astype(np.float32), b.astype(np.float32) # float32's work in theano and tensorflow without them complaining to us


# used for convolutional neural networks
"""shape will be a tuple of 4 different values."""
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

# calculate the cross entropy from the definition for sigmoid cost (used for binary classification)
def sigmoid_cost(T, Y):
    return -(T * np.log(Y) + (1 - T) * np.log(1 - Y)).sum()

# more general cross entropy cost function which we use for softmax
def cost(T, Y):
    return -(T * np.log(Y)).sum()

"""where as the function cost is direct from the definition, cost2 also
calculates the softmax cross entropy, but does it in a more fancy way.
so here we only use the actual values where the targets are non-zero
note that cost and cost2 should both give us the same answer.
cost2 is a little bit more efficient."""
def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

# error rate between the targets and the predictions
def error_rate(targets, predictions):
    return np.mean(targets != predictions)


"""turns an Nx1 vector of targets, which will have the class labels 0 to K-1,
and turns it into an indicator matrix which will only have the values 0 and 1.
But its size will be NxK"""
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

# get all the data from all the classes
def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    # N = 35887
    Y = []
    X = []
    first = True # skip the first line, since its just the headers
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0])) # we know that the first column is the label
            X.append([int(p) for p in row[1].split()]) # the second column are space seperated pixels

    # convert them into numpy arrays and normalize the data
    X, Y = np.array(X) / 255.0, np.array(Y)

    """since our classes are imbalanced, we lengthen class 1 by repeating it 9
    times. we take all the data thats not in class 1 and stick them into variables
    called X0 and Y0.
    we set X1 equal to be the samples where y is equal to 1(label is class 1)
    we repeat it 9 times
    we then stack X1 and X0 back together
    we then stack Y0 and Y1"""
    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1] * len(X1)))

    return X, Y


"""a function we'll use when talking about convolutional neural networks.
this function, keeps the original image shape."""
def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

"""does almost the same thing as getdata() except that here we only add the
samples for which the class is 0 or 1."""
def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)

# k fold cross validation
def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) / K
    errors = []
    for k in range(K):
        xtr = np.concatenate([X[:k * sz, :], X[(k * sz + sz):, :]])
        ytr = np.concatenate([Y[:k * sz], Y[(k * sz + sz):]])
        xte = X[k * sz:(k * sz + sz), :]
        yte = Y[k * sz:(k * sz + sz)]

        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:", errors)
    return np.mean(errors)
