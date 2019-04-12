import numpy as np


class TD:

    def __init__(self, n):

        self.n = n
        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def get_value(self, x):

        return np.dot(self.w, x)

    def update(self, x, r, xp, alpha, gm, gm_p, lm):

        delta = r + gm_p*np.dot(self.w, xp) - np.dot(self.w, x)
        self.z = x + gm*lm*self.z
        self.w += alpha*delta*self.z
        return delta

    def reset(self):

        self.w = np.zeros(self.n)
        self.z = np.zeros(self.n)

    def reset_et(self):

        self.z = np.zeros(self.n)
