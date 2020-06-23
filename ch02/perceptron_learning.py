import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



class Perceptron():
    def __init__(self):
        self.max_iter=1000
        self.eta=0.1

    def fit(self,x,y_gt):
        x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
        self.w_=np.random.random(x_1.shape[-1])
        for iter in range(self.max_iter):
            y_pred=self._pred(x_1)
            # print('iteration:',iter,y_gt-y_pred,(y_gt-y_pred)[:,np.newaxis])
            dw_=self.eta*np.sum(np.broadcast_to((y_gt-y_pred)[:,np.newaxis],x_1.shape)*x_1,axis=0)
            print(dw_)
            if np.sum(abs(dw_))<10e-9:
                break
            self.w_+=dw_

    def _pred(self,x):
        y_pred = np.sum(x * self.w_, axis=1)
        y_pred = np.array([1 if i > 0 else -1 for i in y_pred])
        return y_pred

    def predict(self,x):
        x = np.hstack((x,np.ones((x.shape[0],1))))
        y_pred = self._pred(x)
        return y_pred


class Adaline():
    def __init__(self,eta=0.0001,n_iter=1000):
        self.max_iter=n_iter
        self.eta=eta
        self.w_initialized = False
        self.shuffle = True

    def fit(self,x,y_gt):
        x_1 = np.hstack((x, np.ones((x.shape[0], 1))))
        self.w_=np.random.random(x_1.shape[-1])
        self.cost_ = []
        for iter in range(self.max_iter):
            hidden=self.net_input(x_1)
            y_pred=self.activation(hidden)
            # dw_=self.eta*np.sum(np.broadcast_to((y_gt-y_pred)[:,np.newaxis],x_1.shape)*x_1,axis=0)
            dw_=self.eta*np.dot(x_1.T ,y_gt-y_pred)
            print(dw_)
            if np.sum(abs(dw_))<10e-9:
                break
            self.w_+=dw_
            self.cost_.append(np.sum((y_gt-y_pred)**2)/2)

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
        return np.array([1 if i > 0 else -1 for i in y_pred])

    def net_input(self,x):
        return np.dot(x,self.w_)#np.sum(x * self.w_, axis=1)

    def activation(self, X):
        return X

    def _initialize_weights(self, m):
        self.w_ = np.random.random(1 + m)
        self.w_initialized = True

    def _shuffle(self, X, y):
        r = np.random.permutation(len(y))
        return X[r], y[r]

def generate_fake_data():
    c = Perceptron()
    num_sample_for_each_cls=10
    x0=np.vstack((np.random.normal(10,.5,num_sample_for_each_cls),
                 np.random.normal(8,.5,num_sample_for_each_cls),
                 np.random.normal(-1,.5,num_sample_for_each_cls),
                 np.random.normal(2,.5,num_sample_for_each_cls))).T
    x1=np.vstack((np.random.normal(0,.5,num_sample_for_each_cls),
                 np.random.normal(2,.5,num_sample_for_each_cls),
                 np.random.normal(5,.5,num_sample_for_each_cls),
                 np.random.normal(5,.5,num_sample_for_each_cls))).T
    print(np.random.normal(10,.5,num_sample_for_each_cls).shape,x1.shape)
    x=np.vstack((x0,x1))
    print(x.shape)
    y=np.hstack((np.ones(num_sample_for_each_cls),np.ones(num_sample_for_each_cls)*-1))
    print(y.shape)
    c.fit(x,y)

    x0_pred=np.vstack((np.random.normal(10,.5,num_sample_for_each_cls),
                 np.random.normal(8,.5,num_sample_for_each_cls),
                 np.random.normal(-1,.5,num_sample_for_each_cls),
                 np.random.normal(2,.5,num_sample_for_each_cls))).T
    x1_pred=np.vstack((np.random.normal(0,.5,num_sample_for_each_cls),
                 np.random.normal(2,.5,num_sample_for_each_cls),
                 np.random.normal(5,.5,num_sample_for_each_cls),
                 np.random.normal(5,.5,num_sample_for_each_cls))).T
    print(c.predict(np.vstack((x0_pred,x1_pred))))

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=0.02):
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

def test_iris_data():
    csv_add='../data/iris.data'
    df =pd.read_csv(csv_add,header=None,encoding='utf-8')
    print(df.head())
    y = df.iloc[0:100, 4].values
    y = np.where(y == 'Iris-setosa', -1, 1)
    X = df.iloc[0:100, [0, 2]].values
    plt.scatter(X[:50, 0], X[:50, 1],color = 'red', marker = 'o', label = 'setosa')
    plt.scatter(X[50:100, 0], X[50:100, 1],color = 'blue', marker = 'x', label = 'versicolor')
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()


    # ppn = Adaline(eta=0.0001)
    # ppn.fit(X,y)
    # plot_decision_regions(X, y, classifier=ppn)
    # plt.show()

    X_std=X.copy()
    X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
    X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()
    ppn = Adaline(eta=0.01)
    ppn.fit(X_std,y)
    plot_decision_regions(X_std, y, classifier=ppn)
    plt.show()

    ada_gd = Adaline(n_iter=20, eta=0.01)
    ada_gd.fit(X_std, y)
    plot_decision_regions(X_std, y, classifier=ada_gd)
    plt.title('Adaline - Gradient Descent')
    plt.xlabel('sepal length [standardized]')
    plt.ylabel('petal length [standardized]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.plot(range(1, len(ada_gd.cost_) + 1),ada_gd.cost_, marker = 'o')
    plt.xlabel('Epochs')
    plt.ylabel('Sum-squared-error')
    plt.tight_layout()
    plt.show()
test_iris_data()