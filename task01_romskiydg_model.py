
import numpy as np
from sklearn.preprocessing import StandardScaler


def LossFunction(coef, X, y, l1_coef, l2_coef):
    res = (np.sum((X@coef - y)**2)/len(X)
           + l1_coef*np.sum(np.abs(coef))
           + l2_coef*np.sum(coef**2))
    return res


class RandomIterator:
    def __iter__(self):
        return self

    def __init__(self, batch_size, length, limit):
        self.limit = limit
        self.random_packages = np.random.randint(length,
                                                 size=(limit, batch_size))
        self.counter = 0

    def __next__(self):
        if self.counter < self.limit:
            self.counter += 1
            return self.random_packages[self.counter - 1]
        else:
            raise StopIteration


class ElasticNetRegressor(object):
    '''
    Параметры
    ----------
    n_epoch    : количество эпох обучения
    alpha      : градиентный шаг
    batch_size : размер пакета для шага SGD
    delta      : параметр численного дифференцирования
    l1_coef    : коэффициент l1 регуляризации
    l2_coef    : коэффициент l2 регуляризации
    '''
    def __init__(self, n_epoch=1000, batch_size=10, alpha=0.0001, delta=0.01,
                 l1_coef=0.5, l2_coef=0.5):
        assert n_epoch > 0 and batch_size > 0, "invalid parameteres"
        assert delta > 0 and alpha > 0, "invalid parameteres"
        assert l1_coef >= 0 and l2_coef >= 0, "invalid parameteres"
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.alpha = alpha
        self.delta = delta
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef
        self.scaler = StandardScaler()

    def fit(self, X, y):
        assert X.ndim == 2 and y.ndim == 1, "invalid train array (%d)" % X.ndim
        assert len(X) == len(y), "trainX and trainY are mismatched"
        assert len(y) > 0, "dataset is empty"
        X_my = self.scaler.fit_transform(X)
        self.answers_mean = np.mean(y)
        y_my = y - self.answers_mean
        ones = np.ones((len(X_my), 1))
        X_with_ones = np.hstack((ones, X_my))
        iterator = RandomIterator(self.batch_size, len(X), self.n_epoch)
        self.coef_ = np.zeros(X_with_ones.shape[1])
        identity = np.eye(len(self.coef_), len(self.coef_))
        gradient = np.zeros(len(self.coef_))

        for batch_indeces in iterator:
            batch = np.take(X_with_ones, batch_indeces, axis=0)
            batch_answers = np.take(y_my, batch_indeces)
            loss = LossFunction(self.coef_, batch, batch_answers,
                                self.l1_coef, self.l2_coef)
            for i in range(len(self.coef_)):
                loss_increment = LossFunction(
                    self.coef_ + self.delta*identity[i],
                    batch, batch_answers, self.l1_coef, self.l2_coef)
                gradient[i] = (loss_increment-loss) / self.delta
            self.coef_ -= self.alpha * gradient
        self.coef_[0] += self.answers_mean - ((self.scaler.mean_ / self.scaler.scale_)
                                              @ self.coef_[1:len(self.coef_)])
        self.coef_ /= np.hstack(([1], self.scaler.scale_))

    def predict(self, X):
        assert X.ndim == 2, "invalid test array"
        assert X.shape[1] == len(self.coef_) - 1, "invalid test array"
        #X_test = self.scaler.transform(X)
        ones = np.ones((len(X), 1))
        X_with_ones = np.hstack((ones, X))
        prediction = X_with_ones @ self.coef_
        return prediction

    def score(self, y_gt, y_pred):
        assert len(y_gt) == len(y_pred), "invalid arrays"
        R2_score = 1 - (np.sum((y_gt - y_pred)**2)
                        / np.sum((y_gt - np.mean(y_gt))**2))
        return R2_score
