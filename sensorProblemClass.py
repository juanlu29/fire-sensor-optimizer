
import numpy as np
import numba as nb
from numba import prange
import sys

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.core.problem import Problem


@nb.njit(parallel=False)
def numbaLoop(x,objective1,objective2, sorting, arrivals, areas, numberSensors, keep_numberIgnitions, not_keep_numberIgnitions, populationSize,f1,f2,constraint,constraintCounter,keep_ignitions,no_keep_ignitions,constraint_label):

    '''
    This function pre-compiles a function to loop through possible sensor locations to identify at each site which wildfire is detected the fastest according the quantity to be minimized.
    This cuantity is stored to compute the objective function

    detected is an array with size as population number, and it adds when a fire is detected.

    optimizingMagnitude can be either arrival times or burnt area (At detection).

    '''

    for p in range(populationSize):
        # These ignitions contribute to time constraint
        for ii in prange(keep_numberIgnitions):
            i = keep_ignitions[ii]
            s = 0
            while not (x[p,sorting[i,s]]):
                if s == numberSensors-1:
                    break
                else:
                    s += 1 
            if constraint_label == 'detectTime':
                deltaConstraint = arrivals[i,sorting[i,s]]-constraint

                if deltaConstraint>0: # Only if it is positive, constitues a constraint violation
                    constraintCounter[p] = constraintCounter[p]+deltaConstraint*deltaConstraint

            elif constraint_label == 'detectArea':
                deltaConstraint = areas[i,sorting[i,s]]-constraint
                    
                if deltaConstraint>0: # Only if it is positive, constitutes a constraint violation
                    constraintCounter[p] = constraintCounter[p]+deltaConstraint*deltaConstraint

            # Outside the numbaLoop, in the optimization function, times != areas, therefore no problem here
            if objective1 == 'times':
                f1[p] = f1[p] + arrivals[i,sorting[i,s]] # adds all every arrival times per ignition
            elif objective1 == 'areas':
                f1[p] = f1[p] + areas[i,sorting[i,s]] # adds all every arrival times per ignition
            if objective2 == 'areas':
                f2[p] = f2[p] + areas[i,sorting[i,s]] # adds all every arrival times per ignition
            elif objective2 == 'times':
                f2[p] = f2[p] + arrivals[i,sorting[i,s]] # adds all every arrival times per ignition

        # These ignitions do not contribute to time constraint
        for ii in range(not_keep_numberIgnitions):
            #print(ii)
            i = no_keep_ignitions[ii]
            s = 0
            while not (x[p,sorting[i,s]]):
                if s == numberSensors-1:
                    break
                else:
                    s += 1 
            # Outside the numbaLoop, in the optimization function, times != areas, therefore no problem here
            if objective1 == 'times':
                f1[p] = f1[p] + arrivals[i,sorting[i,s]] # adds all every arrival times per ignition
            elif objective1 == 'areas':
                f1[p] = f1[p] + areas[i,sorting[i,s]] # adds all every arrival times per ignition
            if objective2 == 'areas':
                f2[p] = f2[p] + areas[i,sorting[i,s]] # adds all every arrival times per ignition
            elif objective2 == 'times':
                f2[p] = f2[p] + arrivals[i,sorting[i,s]] # adds all every arrival times per ignition
                
    return f1, f2, constraintCounter

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
                 keep_ignitions, # ignition indexes that represent fires that can spread to sensor locations according to time constraint
                 not_keep_igs,   # ignition indexes representing fires that do not spread to sensor locatiosn according to time constraint
                 n_sensor,       # number of sensors to deploy
                 arrivalTimes,   # List of arrival times, the number depends on different physical scenarios explored
                 bAreas,         # List of areas at detection
                 sorting,        # List of sortings, the number depends on different physical scenarios explored
                 npop,
                 nof,
                 objective1,     # String indicating which is objective 1
                 objective2,     # String indicating which is objective 2
                 weights,        # Weigths under which the different scenarios are considered. These are normalized 
                 constraint,     # Constraint can be either number of sensors or time. constraint is a list [A,B] such that A is the constraint type implemented and B the number
                 penalizations,  # Penalizations are for time, areas and speeds respectively
                 HeavyLoop):
        if constraint[1] == 0:
            # No constraint
            super().__init__(n_var=n_sensor, n_obj=2, n_ieq_constr=0, xl=0, xu=1, vtype=bool)
        else:
            super().__init__(n_var=n_sensor, n_obj=2, n_ieq_constr=len(constraint[1::]), xl=0, xu=1, vtype=bool)
        

        # Important to regulate the number of constraints
        #n_ieq_constr=2

        self.numberSensors = n_sensor # This parameter cannot change, as it defines the dimensionality of the solution space
        self.weights = np.asarray(weights)
        self.numberScenarios = len(self.weights)

        self.keep_ignitions= [np.asarray(keep_ignitions[i]).astype(int) for i in range(self.numberScenarios)]
        self.not_keep_igs = [np.asarray(not_keep_igs[i]).astype(int) for i in range(self.numberScenarios)] 

        # Optimizing is a string that tells what the problem is optimizing, for example areas or arrival times
        self.objective1 = objective1
        self.objective2 = objective2

        self.arrival = arrivalTimes
        self.areas = bAreas

        # Remember magnitudes to minimize are given in a list. Therefore, for each element the matrix is potentially different
        self.numberIgnitions = np.asarray([self.arrival[j].shape[0] for j in range(self.numberScenarios)])

        # In this case a global penalization is necessary
        if self.objective2 == 'sensors':
            if self.objective1 == 'times':
                self.penalizationGlobal = np.sum(self.numberIgnitions)*penalizations[0] # 5 hours
            elif self.objective1 == 'areas':
                self.penalizationGlobal = penalizations[1] # weight in with weather frequencies sum of max aras in simulations
        else:
            print("objective 2 not properly set")
            sys.exit()

        self.sorting = sorting # Always refer to the time ordering of sensors that detect the wildfire


        # This implementation of the Pymoo Problem Class evaluates the objective function over the whole solution
        # population at a given generation, that is why it is necessary to discriminate between population size and
        # offspring size because the system will calculate depending wheter it is the initial generation or the following
        # ones. Check Pymoo library to understand this better; https://pymoo.org/interface/problem.html
        self.npop = npop
        self.nof = nof

        self.f1_p = np.zeros(npop)
        self.f1_o = np.zeros(nof)
        self.detected_p = np.zeros(npop)
        self.detected_o = np.zeros(nof)

        self.constraint = constraint # The constrain of the optimization is to retrieve a given number the solutions at the pareto front
        if len(self.constraint) == 1:
            self.constraint.append(0) # no constraints, therefore just appending zero to avoid errors later.

        # To increase the computation speed, the core loop of objective function is provided as an numbe pre-compiled input function
        self.heavyLoop = HeavyLoop


    def _evaluate(self, xo, out, *args, **kwargs):
        
        
        x = xo.copy()
        
        populationSize = x.shape[0]
        if populationSize == self.npop:
            f1 = self.f1_p.copy()
            f2 = self.f1_p.copy()
            fb = self.f1_p.copy()
            fb2 = self.f1_p.copy()
        else:
            f1 = self.f1_o.copy()
            f2 = self.f1_o.copy()
            fb = self.f1_o.copy()
            fb2 = self.f1_o.copy()
            
        integerX = np.zeros((populationSize,self.numberSensors),dtype=int)
        integerX[x==True] = 1
        
        # This loop separates f1 and fb because heavyLoop modifies f1 inside, therefore performing a weighted cumulation of the values cannot be used with the same variable.
        out1 = np.zeros(populationSize)
        constraintCounter = np.zeros(populationSize)

        # Checking that bi-objectives are different
        if self.objective1 == self.objective2:
            print("objective 1 and objective 2 cannot be the same")
            sys.exit()
        elif self.objective1 == 'sensors':
            print("objective 1 cannot be minimizing sensors")
            sys.exit()

        for i, weight in enumerate(self.weights):
            f1,f2, constraintCounter = self.heavyLoop(x, self.objective1, self.objective2, self.sorting[i], self.arrival[i], self.areas[i],self.numberSensors, len(self.keep_ignitions[i]), len(self.not_keep_igs[i]), populationSize, f1, f2, self.constraint[1],constraintCounter, self.keep_ignitions[i], self.not_keep_igs[i], self.constraint[0])
            fb = fb + weight*f1
            fb2 = fb2 + weight*f2                                           
            out1 += constraintCounter
            f1[:] = 0.
            f2[:] = 0.
            constraintCounter[:] = 0.
        f1 = fb
        f2 = fb2

        if self.objective2 == 'sensors':
            # The second objective is the sum of sensors
            f2 = np.sum(integerX, axis=1) # integer x does not depend on ignition

            # The next penalization can only happen in case f2 is minimization by sensors, because otherwise solutions cannot be zero due to initial penalizations in arrival times and areas
            f1[np.where(f2==0.)[0]] = self.penalizationGlobal # When f2 objective is null, it is a solution with no sensors. it is necessary to give higher penalization

        # Objective functions output
        out["F"] = np.column_stack([f1, f2])

        # Next statements are constrains
        if self.constraint[0] == 'nSensors':
            out["G"] = self.constraint[1] - np.sum(integerX, axis=1) # Limit the number of sensors explored in the pareto front to the amount solutionsPareto.
        elif (self.constraint[0] == 'detectTime') or  (self.constraint[0] == 'detectArea'):
            if len(self.constraint[1::])>1:
                out2 = np.sum(integerX, axis=1) - self.constraint[2]
                out["G"] = np.column_stack([out1, out2])
            else:
                out["G"] = out1

        return out["F"]