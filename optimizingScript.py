# Importing problem class
import numpy as np
import os
import matplotlib.pyplot as plt
import sensorProblemClass as so

def sortingComputation(nig,ns,magnitude2sort): 
    '''
    This function takes as main input the matrix of the quantity to be minimized. The matrix has the following structure; values at each sensor location (column indexes) corresponding to each wildfire (row indexes).
    Per wildfire (matrix row) the function sorts the quantities from smaller to larger. In practicality this sorting lists the sensor locations that detects the lowest value first, the second next, and so on.
    As the wildfire spreads, the closest sensors start detecting the magnitude. In a context of optimizing sensor locations, this sorting would represent the contribution of the best sensor detecting a particular wildfire.
    It is necessary to provide not only the minimum but the ordered list of sensor locations because during the optimization process the different solutions have most of the sensor sites empty.

    Equally penalized sensor sites do not play a major role in this sorting, therefore it is not relevant which one the sorting algorithm picks the first from those. If a wildfire is not detected in a sensor
    configuration the contribution in the objective functions will be far from optimum.

    The output is a matrix whose entries are integers representing the column locations that sorts the values per row

    for example, if optimizing wildfire arrival times

    for wildfire "n"; sensor 1 detects 100 seconds, sensor 2 detects 20 seconds, sensor 3 detects 75 seconds

    The output would be

    for wildfire "n"; 1,2,0 (in pythonic indexation standard)

    '''

    orderedSensor2Detect = np.zeros((nig,ns),dtype=int)

    for i in range(nig):
        # Deriving indexes
        orderIndexes = np.argsort(magnitude2sort[i,:]) # 1 is added because at the end, zeros are filtered during optimization process
        orderedSensor2Detect[i,:] = orderIndexes

    return orderedSensor2Detect



def burnedAtSensorLocation(fires,validIgnitions,validSensors,nx,ny):
    '''
    To optimize sensor location by minimizing the burnt area of the wildfire at detection,
    the burned areas need to be calculated when a wildfire first spreads into a possible sensor location
    This function computes that a return a matrix with the following size,

    # ignitions x # sensor locations

    Inputs of the function are the ignitions, sensor locations and the wildfire arrival times.

    '''


    bAreas = np.zeros((len(validIgnitions),len(validSensors)),dtype=int)

    # To compute area
    onesLandscape = np.ones(nx*ny)
            
    # For each ignition
    for i, ig in enumerate(validIgnitions):
        # The are possible sensor locations
        for s, ns in enumerate(validSensors):
            if fires[i,ns]>0.:

                print(f'Burned area sensed by sensor {s}, {ns} from ignition {i}, {ig}')

                fire_in_sensor = fires[i,ns]
                fireFromIgnition = fires[i,:]
                locationsArea_fireReachesSensor = np.where((fire_in_sensor>fireFromIgnition) & (fireFromIgnition>0.))[0]
                bArea = np.sum(onesLandscape[locationsArea_fireReachesSensor])

                #plt.imshow(fires[i,:].reshape((nx,ny)))
                #plt.scatter(ns/nx,ns%ny)
                #plt.show()
                bAreas[i,s] = bArea

    return bAreas
        

def loadingFires(simDataPath,validIgnitions,nx,ny):
    '''
    The simulation-driven-optimization problem requires to precompute wildfire simulations that are expected to have ignited at the 
    ignition locations from ignitnios.txt file. This function return the arrival times per ignition in a matrix of size,

    # ignitions x arrival time values per pixel

    Simulations are not run to spread fire over the whole landscape, therefore each row is expected to be sparse

    '''

    # Loading fire data
    names = os.listdir(simDataPath)
    print(f'The are {len(names)} fires in the sim database')
    ignitions = np.zeros(len(names),dtype=int)

    fires = np.zeros((len(validIgnitions),nx*ny))
    for i,sampleIgn in enumerate(validIgnitions.ravel()):
        print(f'fire {sampleIgn}')
        fire = np.loadtxt(simDataPath+f'{sampleIgn}.csv',delimiter=',')
        ignitions[names.index(f'{sampleIgn}.csv')] = fire[-1]
        fires[i,:] = fire[0:-1]

    return fires

# Domain pixel size
nx = 321
ny = 321

# simulation data path
simDataPath = f'./test-data/'

# Loading fire and sensor locations given the Cocentaina case-study
validIgnitions = np.loadtxt('ignitions.txt').astype(int)
nig = len(validIgnitions)
validSensors = np.loadtxt('sensors.txt').astype(int)
ns = len(validSensors)


# Uncomment if fires/bAreas file is missing to recompute it

#print("Computing simulated arrival times data")
fires = loadingFires(simDataPath,validIgnitions,nx,ny)
# Limiting fire arrival times at possible sensor locations
firesAtSensors = fires[:,validSensors]
np.save('firesAtSensors',firesAtSensors,allow_pickle=True) # For this case study this file is about 3.7 Gb of data.
print("Loading simulated arrival times data")
firesAtSensors = np.load('firesAtSensors.npy',allow_pickle=True)

#print("Computing burned areas at sensor locations")
bAreas = burnedAtSensorLocation(fires,validIgnitions,validSensors,nx,ny)
np.save('bAreas',bAreas,allow_pickle=True)
print("Loading burned areas at sensor locations")
bAreas = np.load('bAreas.npy',allow_pickle=True)


# Preparing data for optimization. Since data is sparse, during optimization the computation of objective functions may find as minimum null values, which obscure the optimization process.
# This issue is solved by penalizing those pixels without wildfire presence with maximal values.
firesAtSensors[np.where(firesAtSensors==0.)] = 5*3600 # Arrival times are extrapolated to the maximum simulated value (5 hours in this case, which is expressed in seconds, the units in which arrival times are presented)
bAreas[np.where(bAreas==0.)] = np.amax(bAreas)    # Burned areas are extrapolated to the maximum burnt area of the whole wildfire dataset dataset 


# Choose optimization by times or burned areas
optimizeBy = 'times'
print('Optimizing by '+optimizeBy)
if optimizeBy == 'times':

    # Fast optimization requires to pre-sort the different arrival times / burned areas from the different wildfires detected by each sensor. Wildfires do not spreading over a given sensor have their corresponding sensor location pixel penalized and do not matter and their sorting has no role.
    indexes =  sortingComputation(nig,ns,firesAtSensors)
    magnitude = firesAtSensors

elif optimizeBy == 'bAreas':

    # Fast optimization requires to pre-sort the different arrival times / burned areas from the different wildfires detected by each sensor. Wildfires do not spreading over a given sensor have their corresponding sensor location pixel penalized and do not matter and their sorting has no role.
    indexes =  sortingComputation(nig,ns,bAreas)
    magnitude = firesAtSensors


# The pattern of the matrixes is characteristic. Plot of their values to be sured that this process has been done accordingly
plt.figure()
plt.title(optimizeBy+' at sensors')
plt.imshow(firesAtSensors)
plt.imshow(bAreas)
plt.show()

plt.figure()
plt.title('Sorted indexes pattern')
plt.imshow(indexes)
plt.show()


# This optimization methodology admits the weigthed contribution of several simulated scenarios at once.
# For this test-case, the same set of simulated fires with the same wind conditions is used to mimic a multiple-weigthed scenario.
# The aim is to take into account wildfire simulations computed using the different eight wind cardinal directions.
# Each wind direction is weighted by the frequency of the wind rose chart and the input wind speed is a value representative during high fire risk conditions.
weighedScenario =  np.zeros((8,indexes.shape[0],indexes.shape[1]))
weightedIndexes = np.zeros((8,indexes.shape[0],indexes.shape[1]),dtype=int)
for i in range(8):
    weighedScenario[i,:,:] = magnitude
    weightedIndexes[i,:,:] = indexes

# Equally distributed weights
weights = np.ones(8)/8

# Population values for optimization
npop = 100 # Total population of solutions
solutionsPareto = npop # Constrains how many solutions at the pareto front will be explored and returned.
nOffsprings = 20 # Number of offsprings during evolutionary generation

# Instantiating Problem classes
problem = so.optimalSensorLocWeigthed(ns,weighedScenario,weightedIndexes,npop,nOffsprings,optimizeBy,weights,solutionsPareto,so.numbaLoop)

# Finally, computation of the pareto front solutions

nTermination = 10000
algorithm = so.NSGA2(
   pop_size=npop,
    n_offsprings=nOffsprings,
    sampling=so.BinaryRandomSampling(),
    crossover=so.TwoPointCrossover(),
    mutation=so.BitflipMutation(),
    eliminate_duplicates=True
)

# During minimizating, it takes many generation before the constrained number of pareto solutions is found. Otherwise F, and X results will be None.
termination = so.get_termination("n_gen", nTermination)
res = so.minimize(problem,
            algorithm,
            termination,
            seed = 1,
            verbose=True)


print("Best solution found: %s" % res.X.astype(int))
print("Function value: %s" % res.F)
print("Constraint violation: %s" % res.CV)
Xt = res.X
Ft = res.F

# calculate a hash to show that all executions end with the same result
print("hash", res.F.sum())

plt.figure(figsize=(10, 10))
plt.scatter( res.F[:, 0],  res.F[:, 1], s=30)
plt.title("Objective Space")
plt.show()

