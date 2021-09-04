import numpy as np
import matplotlib.pylab as plt


class KalmanFilter(object):
    def __init__(self, initial_position=(0, 0)):
        self.delta_t = 1
        self.P = np.identity(4)
        self.Q = np.identity(4)
        self.R = np.identity(2)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.A = np.identity(4)
        self.A[0, 2] = self.delta_t
        self.A[1, 3] = self.delta_t
        self.A_predict = np.identity(4)
        self.A_predict[0, 2] = self.delta_t / 2
        self.A_predict[1, 3] = self.delta_t / 2

        self.x_before = np.array([[initial_position[0], initial_position[1], 0, 0]])
        self.x_before = self.x_before.transpose()

        # Initial current priori state estimate and posteriori state estimate
        self.pre_before = None
        self.pre_current = None
        self.cor_current = None
        self.save_list = (0, 1)

    def run(self, new_data, show_priori=False, show_posteriori=False):
        # Predict
        x_pre = self.predict(mode=1)
        P_pre = np.dot(np.dot(self.A, self.P), self.A.transpose()) + self.Q

        # Correct
        k1 = np.dot(P_pre, self.H.transpose())
        k2 = np.dot(self.H, k1) + self.R
        K = np.dot(k1, np.linalg.inv(k2))

        z = np.array([new_data]).transpose()
        x_cor = x_pre + np.dot(K, z - np.dot(self.H, x_pre))
        self.P = np.dot(np.identity(self.P.shape[0]) - np.dot(K, self.H), P_pre)
        self.x_before = x_cor

        if show_priori:
            print(x_pre.transpose())
        if show_posteriori:
            print(x_cor)
        self.cor_current = [x_cor[i, 0] for i in self.save_list]

        return self.pre_current, self.cor_current

    def predict(self, mode=0, replace=False):
        # Predict
        if mode == 0:
            x_pre = np.dot(self.A_predict, self.x_before)
        else:
            x_pre = np.dot(self.A, self.x_before)

        # replace x_before by predict value
        if replace:
            self.x_before = x_pre

        self.pre_before = self.pre_current
        self.pre_current = [x_pre[i, 0] for i in self.save_list]
        return x_pre

    def get_priori_estimate(self, reset=True, t=0):
        if t == 0:
            z_pri = self.pre_current
            if reset:
                self.pre_current = None
        elif t == -1:
            z_pri = self.pre_before
            if reset:
                self.pre_before = None
        return z_pri

    def get_posteriori_estimate(self):
        z_cor = self.cor_current
        self.cor_current = None
        return z_cor


def show(samples, pre_list, cor_list, show_priori=True, show_posteriori=True):
    x = [samples[i][0] for i in range(len(samples))]
    y = [samples[i][1] for i in range(len(samples))]
    plt.plot(x, y, label='Actual Track')
    plt.scatter(x, y)

    if show_priori:
        x = [round(pre_list[i][0], 3) for i in range(len(pre_list))]
        y = [round(pre_list[i][1], 3) for i in range(len(pre_list))]
        plt.plot(x, y, 'r', label='Priori state estimate')
        plt.scatter(x, y, c='r')

    if show_posteriori:
        x = [round(cor_list[i][0], 3) for i in range(len(cor_list))]
        y = [round(cor_list[i][1], 3) for i in range(len(cor_list))]
        plt.plot(x, y, 'g', label='Posteriori state estimate')
        plt.scatter(x, y, c='g')

    plt.legend()
    plt.show()


if __name__ == "__main__":
    t = np.linspace(0, 10, 100)
    x = np.sin(t)
    x_pre_list = []
    x_cor_list = []
    x_samples = []
    my_filter = KalmanFilter(initial_position=[t[0], x[0]])
    import time
    t0 = time.time()
    for i in range(len(t)):
        a, b = my_filter.run([t[i], x[i]])
        x_samples.append([t[i], x[i]])
        x_pre_list.append(a)
        x_cor_list.append(b)
        #my_filter.predict()
    for i in range(10):
        pass
        #my_filter.run(my_filter.x_cor_list[-1])
    t = time.time() - t0
    print(t)
    show(x_samples, x_pre_list, x_cor_list, show_posteriori=False)
