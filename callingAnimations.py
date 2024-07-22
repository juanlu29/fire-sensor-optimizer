'''
This script generate the animations of the pareto convergence.
It is necessary to modify the list paretos to include the files to compute animations
'''
import numpy as np
import pickle
import make_animations as ma
import sensorFunctions as sf
from pymoo.core.callback import Callback
import sys


class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["n_iter"] = []
        self.data["sols"] = []

    def notify(self, algorithm):
        if algorithm.n_iter <= 10: 
            self.data["best"].append(algorithm.pop.get("F"))
            self.data["n_iter"].append(algorithm.n_iter)
            self.data["sols"].append(algorithm.pop.get("X"))
        if (algorithm.n_iter > 10) and (algorithm.n_iter < 500):
            if algorithm.n_iter%50 == 0:
                self.data["best"].append(algorithm.pop.get("F"))
                self.data["n_iter"].append(algorithm.n_iter)
                self.data["sols"].append(algorithm.pop.get("X"))     
        if (algorithm.n_iter > 500) and (algorithm.n_iter < 1000):
            if algorithm.n_iter%100 == 0:
                self.data["best"].append(algorithm.pop.get("F"))
                self.data["n_iter"].append(algorithm.n_iter)
                self.data["sols"].append(algorithm.pop.get("X"))  
        if (algorithm.n_iter > 1000): 
            if algorithm.n_iter%500 == 0:
                self.data["best"].append(algorithm.pop.get("F"))
                self.data["n_iter"].append(algorithm.n_iter)
                self.data["sols"].append(algorithm.pop.get("X"))

# Case input
case = int(sys.argv[1]) # Input case to compute appropriate animations

# Necessary data

# Loading fire and sensor locations given the Cocentaina case-study
validIgnitions = np.loadtxt('ignitions.txt').astype(int)
nig = len(validIgnitions)
validSensors = np.loadtxt('sensors.txt').astype(int)
ns = len(validSensors)

# Penalizations
penalizations = [5*3600,100*100,0] # Maybe the penalization must be the largest value of the area of interest. This way, small fires have less penalization...

objectives1 = ['areas'] #,'times','times','times','times'] #,'areas','areas','areas','areas']#,['times','areas','speeds','times','areas','speeds']
objectives2 = ['sensors'] #,'sensors','sensors','sensors','sensors'] #,'areas','areas','areas','areas']#,['times','areas','speeds','times','areas','speeds']

# constraints
constraints = [['nSensors',400]]
nTerminations = [500000]
npops = [300]


firesAtSensors = [] # np.zeros((8,nig,ns))
bAreas = [] # np.zero s((8,nig,ns))
speedAtSensors = []
nigs = []
keep_igs = []
not_keep_igs = []
for i in range(8):
    print("Loading simulated arrival times data")
    firesData = np.load(f'firesAtSensors_{i}.npy',allow_pickle=True)

    print("Loading burned areas at sensor locations")
    bAreasData = np.load(f'bAreas_{i}.npy',allow_pickle=True)

    firesAtSensors.append(firesData)
    bAreas.append(bAreasData)

    # Preparing data for optimization. Since data is sparse, during optimization the computation of objective functions may find as minimum null values, which obscure the optimization process.
    # This issue is solved by penalizing those pixels without wildfire presence with maximal values.
    firesAtSensors[-1][np.where(firesAtSensors[-1]==0.)] = penalizations[0] # Arrival times are extrapolated to the maximum simulated value (5 hours in this case, which is expressed in seconds, the units in which arrival times are presented)
    # Penalization areas is different, as each fire has different area.
    # Area penalization is the largest area per fire.
    maxAreas = np.loadtxt(f'maxAreas_{i}')
    for ji in range(len(bAreas[-1])):
        bAreas[-1][ji,np.where(bAreas[-1][ji,:]==0.)] = maxAreas[ji]    # Burned areas are extrapolated to the maximum burnt area of the whole wildfire dataset dataset

# Sorting
indexes = []
for i in range(8):
    # Fast optimization requires to pre-sort the different arrival times / burned areas from the d
    indexes.append(sf.sortingComputation(nig,ns,firesAtSensors[i])) # Now we are not omitting any fire

paretos = ['pickle_areas_sensors_c_nSensors_v_400_500000_npop_300_ftol_3.4304131629078194e-08.pkl',]

if case == 1:

    for npop, penalization, objective1, objective2, constraint, nTermination, pareto in zip(npops, penalizations, objectives1, objectives2, constraints, nTerminations, paretos):
        with open(pareto, 'rb') as file:
            res = pickle.load(file)
        print('Processing animation for pareto data from {pareto}')
        ma.animCase1(res, npop, ns, nig, validIgnitions, indexes, firesAtSensors, bAreas, penalizations,objective1,objective2,constraint,nTermination)

if case == 2:

    # Load the object from the saved file
    res_list = []
    for pareto in paretos:
        with open(pareto, 'rb') as file:
            res_list.append(pickle.load(file))

    # Making animation
    ma.animCase2(res_list, npops[0], ns, nig, validIgnitions, indexes, firesAtSensors, bAreas, penalizations[0],objectives1[0],objectives2[0],constraints,nTerminations[0])