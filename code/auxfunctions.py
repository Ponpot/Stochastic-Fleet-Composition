
import numpy as np
from numba import njit

#we use numba decor to accelarate the computation
@njit
def combination(n: int, k: int) -> int:
    if k > n:
        return 0

    numerator = 1
    denominator = 1

    for i in range(k):
        numerator *= (n - i)
        denominator *= (i + 1)

    return numerator // denominator


class AuxFunc():

    @njit
    def prob_k(pimax: int, pij: float) -> np.array:
        """Generate probability.

        Args:
            pimax: int - maximum number of vehicles
            pij: float in [0, 1] - parameter for distribution

        Returns:
            float in (0, 1) - probability at time j, there are k vehicles of type i required
        """
        assert 0 <= pij <= 1
        
        prob = np.zeros(pimax)
        for k in range(pimax):
            #prob[k] = comb(pimax, k) * pij**k * (1-pij)**(pimax - k)
            prob[k] = combination(pimax,k) * pij**k * (1-pij)**(pimax - k)

        return prob

    @njit
    def get_prob(i: int, m: int, pimax: int, rho: np.array) -> np.array:
        """Generate probability.

        Args:
            n: int - number of vehicle types
            m: int - number of time periods
            pmax: np.array - maximum number of vehicles of each type
            rho: np.array(float) in [0,1] - parameter for distribution

        Returns:
            np.array - probability at time j, there are k vehicles of type i required
        """

        probi = np.zeros((m,pimax))
        for j in range(m):
            for k in range(pimax):
                probi[j,k] = abs(combination(pimax,k) * (rho[i,j]**k) * ((1-rho[i,j])**(pimax - k)))
            

        return probi

    @njit
    def get_value_fi(beta_i:float, gamma_i: float, prob_i: np.array, pimax: float, m: int, xi: float, i: int) -> float:

        # get cost for each stage j
        value_j = np.array([np.sum(np.array([ prob_i[j,k] 
                                            *(beta_i * min(k, xi) 
                                            + gamma_i * max(k - xi, 0)) 
                                            for k in range(pimax)]))
                                            for j in range(m)])

                
        return np.sum(value_j)
