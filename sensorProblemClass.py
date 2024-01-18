
import numpy as np
import numba as nb
from numba import prange

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import Problem


@nb.njit(parallel=True)
def numbaLoop(x, sorting, optimizingMagnitude, numberSensors, numberIgnitions, minimumTimes, populationSize,f1):

    '''
    This function pre-compiles a function to loop through possible sensor locations to identify at each site which wildfire is detected the fastest according the quantity to be minimized.
    This cuantity is stored to compute the objective function

    optimizingMagnitude can be either arrival times or burnt area (At detection).

    '''

    minimumTimes[:] = 0.
    for p in prange(populationSize):
        #minimumTimes = np.zeros(self.numberIgnitions)
        for i in prange(numberIgnitions):
            s = 0
            while not (x[p,sorting[i,s]]):
                if s == numberSensors-1:
                    break
                else:
                    s += 1
            f1[p] = f1[p] + optimizingMagnitude[i,sorting[i,s]] # adds all every arrival times per ignition
        
    return f1

class optimalSensorLocWeigthed(Problem):

    '''
    This Pymoo Problem Class optimizes the sensor locations minimizing a physical magnitude at fire,
    either burned area or arrival time. This implementation assumes detection is confined exclusively
    to the pixel where the sensor is deployed.

    In this specific implementation, the optimization process admits different environmental conditions
    in the format of different sets of fire simulations. For instance, to consider the eight different
    wind directions, it is necessary to input a list of arrival time matrixes for each wind direction explored.
    Altogether is necessary to provide for a set of normzalized weights (for instance their normalized frequency),
    so that each contribution to the objective function is accordingly weighted. 

    numbaLoop is a "just in time"  (jit) compiled by numba function that loops through every possible sensor location to find
    the minimum magnitude to be minimized.
    '''

    def __init__(self,
                 n_sensor,   # number of sensors to deploy
                 arrivalTimes,  # List of arrival times, the number depends on different physical scenarios explored
                 sorting,       # List of sortings, the number depends on different physical scenarios explored
                 npop,
                 nof,
                 optimizing, 
                 weights,       # Weigths under which the different scenarios are considered. These are normalized
                 solutionsPareto, 
                 HeavyLoop):
        
        super().__init__(n_var=n_sensor, n_obj=2, n_ieq_constr=1, xl=0, xu=1, vtype=bool)
                         

        # Important to regulate the number of constraints
        n_ieq_constr=1

        self.weights = np.asarray(weights)
        self.numberScenarios = len(self.weights)

        # Optimizing is a string that tells what the problem is optimizing, for example areas or arrival times
        self.optimizing = optimizing

        self.arrivalTimes = arrivalTimes
        self.sorting = sorting

        # Remember now shape has 3 components. First one is for the number of scenarios explored
        self.numberIgnitions = self.arrivalTimes.shape[1]
        self.numberSensors = self.arrivalTimes.shape[2]
        
        if self.optimizing == "areas":
            self.penalizationGlobal = self.numberSensors*100*100 # Approximately areas of 100x100
        elif self.optimizing == "times":
            self.penalizationGlobal = self.numberSensors*5*3600 # 5 hours

        # This implementation of the Pymoo Problem Class evaluates the objective function over the whole solution
        # population at a given generation, that is why it is necessary to discriminate between population size and
        # offspring size because the system will calculate depending wheter it is the initial generation or the following
        # ones. Check Pymoo library to understand this better; https://pymoo.org/interface/problem.html
        self.npop = npop
        self.nof = nof

        self.f1_p = np.zeros(npop)
        self.f1_o = np.zeros(nof)

        self.minimumTimes = np.zeros(self.numberIgnitions)

        self.solutionsPareto = solutionsPareto # The constrain of the optimization is to retrieve a given number the solutions at the pareto front

        # To increase the computation speed, the core loop of objective function is provided as an numbe pre-compiled input function
        self.heavyLoop = HeavyLoop

    
    def _evaluate(self, xo, out, *args, **kwargs):
        
        
        x = xo.copy()

        
        populationSize = x.shape[0]
        if populationSize == self.npop:
            f1 = self.f1_p.copy()
            fb = self.f1_p.copy()
        else:
            f1 = self.f1_o.copy()
            fb = self.f1_o.copy()
            
        integerX = np.zeros((populationSize,self.numberSensors),dtype=int)
        integerX[x==True] = 1
        
        # This loop separates f1 and fb because heavyLoop modifies f1 inside, therefore performing a weighted cumulation of the values cannot be used with the same variable.
        for i, weight in enumerate(self.weights):
            fb = fb + weight*self.heavyLoop(x, self.sorting[i,:,:], self.arrivalTimes[i,:,:],self.numberSensors, self.numberIgnitions, self.minimumTimes, populationSize,f1)
            f1[:] = 0.
        f1 = fb

        # The second objective is the sum of sensors
        f2 = np.sum(integerX, axis=1) # integer x does not depend on ignition
        f1[np.where(f2==0.)[0]] = self.penalizationGlobal # When f2 objective is null, it is a solution with no sensors. it is necessary to give higher penalization

        # Objective functions output
        out["F"] = np.column_stack([f1, f2])

        # Next statements are constrains
        out["G"] = np.sum(integerX, axis=1) - self.solutionsPareto # Limit the number of sensors explored in the pareto front to the amount solutionsPareto.
        return out["F"]