
import numpy as np
from scipy.special import comb
from docplex.mp import model
from auxfunctions import AuxFunc

class Algorithms:
    """Args:
    data: input data for the models
    timeout: time limits for each models, default 200s
    """
    def __init__(self, data, timeout: float = 200) -> None:
        paramter = data.get_parameters()
        rho = paramter[4]
        pmax = paramter[3]
        self.data = data
        self.m = data.m
        self.n = data.n
        self.pmax = pmax
        self.c = paramter[5]
        self.b = paramter[6]
        self.alpha = paramter[0]
        self.beta = paramter[1]
        self.gamma = paramter[2]
        self.rho = rho
        self.timeout = timeout
        self.prob = [AuxFunc.get_prob(i,data.m,pmax[i],rho) for i in range(self.n)]

    def get_value_fi(self, xi: float, i: int) -> float:
        """Generate cost value of type i

        Args:
            x: current selection
            i: vehicle type i

        Returns:
            float: cost of maintenance and hiring more vehicles
        """
                
        return AuxFunc.get_value_fi(self.beta[i],self.gamma[i],self.prob[i],self.pmax[i],self.m,xi,i)
    
    def get_value(self, x: np.array) -> float:
        """Generate cost value

        Args:
            x: current selection

        Returns:
            float: cost of maintenance and hiring more vehicles
        """
        
        return np.sum(np.array([self.get_value_fi(x[i], i)
                        for i in range(self.n)]))

    
    def get_sub_gradient(self, x: np.array) -> np.array:
        """Generate subgradient

        Args:
            x: current selection

        Returns:
            np.array: an array of subgradient vector at x
        """
        subgrad = np.zeros(self.n)
        for i in range(self.n):
            if x[i].is_integer():
                
                if x[i] < 1:
                    subgrad[i] = self.get_value_fi(x[i]+1,i) - self.get_value_fi(x[i],i)
                else:
                    subgrad[i] = 1/2*(self.get_value_fi(x[i]+1,i) - self.get_value_fi(x[i]-1,i))
                    
            else:
                subgrad[i] = self.get_value_fi(np.ceil(x[i]),i) - self.get_value_fi(np.floor(x[i]),i)

        return subgrad

    def linearization(self):

        # set up linear model
        ml = model.Model(log_output=False)
        ml.parameters.timelimit = self.timeout

        # set variables
        x = ml.integer_var_list(self.n, name = "x", ub = self.pmax, lb = 0)

        # set auxiliary variables 
        ijk = [(i,j,k) for i in range(self.n) 
                      for j in range(self.m)
                      for k in range(self.pmax[i])]
        
        y = ml.continuous_var_dict(ijk, name = "y")

        # set up constraints
        ml.add_constraint(ml.sum(self.c[i]*x[i] for i in range(self.n)) <= self.b)

        # add y constraints
        ml.add_constraints(y[i,j,k] >= self.prob[i][j,k]*k*self.beta[i] for (i,j,k) in ijk)
        ml.add_constraints(y[i,j,k] >= self.prob[i][j,k]*(self.beta[i]*x[i] + self.gamma[i]* (k - x[i]))
                          for (i,j,k) in ijk)

        # objective function
        ml.minimize(ml.sum(self.m*self.alpha[i]*x[i] for i in range(self.n))
                            + ml.sum(y[i,j,k] for (i,j,k) in ijk))

        ml.solve()

        return ml.objective_value, ml.solve_details.time, ml.solve_details.mip_relative_gap
    
    def linearization_relaxed(self):

        # set up linear model
        ml = model.Model(log_output=False)
        ml.parameters.timelimit = self.timeout

        # set variables
        x = ml.continuous_var_list(self.n, name = "x", ub = self.pmax, lb = 0)

        # set auxiliary variables 
        ijk = [(i,j,k) for i in range(self.n) 
                      for j in range(self.m)
                      for k in range(self.pmax[i])]
        
        y = ml.continuous_var_dict(ijk, name = "y")

        # set up constraints
        ml.add_constraint(ml.sum(self.c[i]*x[i] for i in range(self.n)) <= self.b)

        # add y constraints
        ml.add_constraints(y[i,j,k] >= self.prob[i][j,k]*k*self.beta[i] for (i,j,k) in ijk)
        ml.add_constraints(y[i,j,k] >= self.prob[i][j,k]*(self.beta[i]*x[i] + self.gamma[i]* (k - x[i]))
                          for (i,j,k) in ijk)

        # objective function
        ml.minimize(ml.sum(self.m*self.alpha[i]*x[i] for i in range(self.n))
                            + ml.sum(y[i,j,k] for (i,j,k) in ijk))

        ml.solve()

        return ml.objective_value, ml.solve_details.time, ml.solve_details.mip_relative_gap


    def cutting_plane(self,iter_max: int = 200, tol: float = 1e-4):

        # set up linear model
        m = model.Model(log_output=False)
        m.parameters.timelimit = self.timeout

        # set variables
        x = m.integer_var_list(self.n, name = "x", ub = self.pmax, lb = 0)

        # set up constraints
        m.add_constraint(m.sum(self.c[i]*x[i] for i in range(self.n)) <= self.b)

        # get initial solution
        m.minimize(m.sum(self.alpha[i]*x[i] for i in range(self.n)))
        
        m.solve()
        xk = np.array(m.solution.get_value_list(x))
        
        # set upperbound
        lower = 0
        upper = lower + 1
        num_iter = 0
        time_solve = 0

        # set theta
        theta = m.continuous_var(name="theta")

        # objective function
        m.minimize(theta + self.m * m.sum(self.alpha[i]*x[i] for i in range(self.n)))

        # start adding cutting planes
        while upper - lower > tol and num_iter < iter_max and time_solve < self.timeout: 

            # get gradient
            dfx = self.get_sub_gradient(xk)


            # update the cutting plane
            m.add_constraint(theta >= m.sum( dfx[i] * (x[i] - xk[i]) for i in range(self.n)) 
                             + self.get_value(xk)
                             )
            
            # solve the new model
            m.solve()
            sol = m.solution 

            # get new vector
            xk = np.array(sol.get_value_list(x))
            
            # update lower bound
            lower = m.objective_value

            # update upperbound
            upper = self.m * sum(self.alpha[i]*xk[i] for i in range(self.n)) + self.get_value(xk)

            # update steps
            num_iter += 1
            
            # update timesolve
            time_solve += m.solve_details.time

            
        optimal = 1 if upper < tol else 0

        return upper, m.objective_value, time_solve, num_iter, optimal
            
