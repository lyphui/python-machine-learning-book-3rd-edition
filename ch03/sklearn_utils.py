
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron,LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


def plot_decision_regions(X, y, classifier,test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],y=X[y == cl, 1],alpha=0.8, c=colors[idx],
                marker=markers[idx], label=cl, edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o', s=100, label='test set')

def load_iris_data():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target
    print('Class labels:', np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1, stratify = y)

    print('Labels counts in y_train:', np.bincount(y_train))
    print('Labels counts in y_test:', np.bincount(y_test))

    sc = StandardScaler()
    sc.fit(X_train) #Using the fit method, StandardScaler estimated the parameters
    # μ (sample mean) and σ (standard deviation) for each feature dimension from the training data.
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std,X_test_std,y_train, y_test

def test_ppn(X_train_std,X_test_std,y_train, y_test):
    ppn = Perceptron(max_iter=40, eta0=0.1, random_state=1)
    ppn.fit(X_train_std, y_train)

    y_pred = ppn.predict(X_test_std)
    print('Misclassified samples: %d' % (y_test != y_pred).sum())
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    plot_decision_regions(X=X_combined_std,y = y_combined,classifier = ppn,test_idx = range(105, 150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.show()

class Logistic_regression():
    def __init__(self,eta=0.0001,n_iter=1000):
        self.max_iter=n_iter
        self.eta=eta
        self.w_initialized = False
        self.shuffle = True
        self.cost_ = []

    def fit(self,x,y_gt):
        x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
        self.w_=np.random.random(x_1.shape[-1])

        for iter in range(self.max_iter):
            hidden=self.net_input(x_1)
            y_pred=self.activation(hidden)
            dw_=self.eta*np.dot(x_1.T ,y_gt-y_pred)
            print(dw_)
            if np.sum(abs(dw_))<10e-9:
                break
            self.w_+=dw_
            self.cost_.append(-(y_gt*(np.log(y_pred)+(1-y_gt)*np.log(1-y_pred))))

    def partial_fit(self, X, y):
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def predict(self,x):
        x = np.hstack((x,np.ones((x.shape[0],1))))
        y_pred = self.activation(self.net_input(x))
        return np.array([1 if i > 0.5 else 0 for i in y_pred])

    def net_input(self,x):
        return np.dot(x,self.w_)#np.sum(x * self.w_, axis=1)

    def activation(self, X):
        return 1/(1+np.exp(-np.clip(X, -250, 250)))

    def _initialize_weights(self, m):
        self.w_ = np.random.random(1 + m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

def plot_sigmoid():
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    z = np.arange(-7, 7, 0.1)
    phi_z = sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$') # y axis ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)
    plt.show()

X_train_std,X_test_std,y_train, y_test= load_iris_data()



weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c, random_state=1,
                            solver='lbfgs',
                            multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)

weights = np.array(weights)
plt.plot(params, weights[:, 0],
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--',
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
#plt.savefig('images/03_08.png', dpi=300)
plt.show()