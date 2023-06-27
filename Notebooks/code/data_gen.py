
import numpy as np

class DataGeneration:
    """Args:
    n: int - number of vehicle types
    m: int - number of time periods
    pm: int - max number of vehicles for total
    """
    def __init__(self, n: int, m: int, pm: int, random_state: int) -> None:

        self.n = n
        self.m = m
        self.pm = pm
        self.random_state = random_state

    def get_parameters(self):
        """Generate parameters.

        Returns:
            alpha: list of random integers between [1, 1000]
            beta: list of random integers between [1, 1000]
            gamma: list of random integers between [alpha[i]+beta[i], 2*(alpha[i]+beta[i])]
            pmax: list of maximum number of vehicles of each type are randon integer between [1,pm]
            rho: n x m array of random floats between [0, 1]
            c: list of random integers between [1, 100] (cost of each type of vehicle)
            b: capacity
        """
        np.random.seed(self.random_state)
        alpha = np.random.randint(2, 100, size=self.n)
        beta = np.random.randint(2, 100, size=self.n)
        gamma = np.random.randint((alpha + beta)+1, 2 * (alpha + beta))
        pmax = np.random.randint(int(self.pm/2), self.pm , size=self.n)
        rho = np.round(np.random.rand(self.n, self.m),2)
        c = np.random.randint(2, 10, size=self.n)
        b_min = int(np.sum(c*pmax)/self.n)
        b = np.random.randint(b_min, int(2*b_min))

        return alpha, beta, gamma, pmax, rho, c, b