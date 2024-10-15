# Importing problem class
import numpy as np
import os
import matplotlib.pyplot as plt
import sensorProblemClass as so
import sensorFunctions as sf
from pymoo.core.callback import Callback
import sys
import pickle
from pymoo.termination.ftol import MultiObjectiveSpaceTermination
from pymoo.termination.robust import RobustTermination
from pymoo.termination.default import DefaultMultiObjectiveTermination

class MyCallback(Callback):

    def __init__(self) -> None:
        super().__init__()
        self.data["best"] = []
        self.data["n_iter"] = []
        self.data["sols"] = []
        self.data["iters"] = []

    def notify(self, algorithm):
        self.data["iters"].append(algorithm.n_iter)
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


def burnedArea2file(fires,j,time):
    # Computes the  area per fire for a given time and saves the information to a txt file

    areas = np.zeros(len(fires),dtype=int)
    i = 0
    for fire in fires:
        areas[i] = len(fire[(fire>0.) & (fire<time)])
        i += 1

    # Write to a file
    np.savetxt(f'maxAreas_wind_{j}_time_{time}',areas,fmt='%d')

    return


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
        # The are possible len(validSensors) possible sensor locations
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


simDataPath1 = f'./sims_p90_N/'
simDataPath2 = f'./sims_p90_NE/'
simDataPath3 = f'./sims_p90_E/'
simDataPath4 = f'./sims_p90_SE/'
simDataPath5 = f'./sims_p90_S/'
simDataPath6 = f'./sims_p90_SW/'
simDataPath7 = f'./sims_p90_W/'
simDataPath8 = f'./sims_p90_NW/'
simDataPaths = [simDataPath1,simDataPath2,simDataPath3,simDataPath4,simDataPath5,simDataPath6,simDataPath7,simDataPath8]

frequencies = [0.073,0.1252,0.1204,0.1333,0.1127,0.1599,0.1859,0.0896]

# Loading fire and sensor locations given the Cocentaina case-study
validIgnitions = np.loadtxt('ignitions.txt').astype(int)
nig = len(validIgnitions)
validSensors = np.loadtxt('sensors.txt').astype(int)
ns = len(validSensors)

# Penalizations
areaGlobalPenalization = 0
for w in range(len(frequencies)):
    maxAreas = np.loadtxt(f'maxAreas_{w}')
    areaGlobalPenalization += np.sum(maxAreas)*frequencies[w]

penalizations = [5*3600,areaGlobalPenalization,0] # Maybe the penalization must be the largest value of the area of interest. This way, small fires have less penalization...

objectives1 = ['areas','areas']# ,'areas','areas','areas','areas','areas','areas','areas']#,'areas','areas']#,'areas','areas']

objectives2 = ['sensors','sensors']# ,'sensors','sensors','sensors','sensors','sensors','sensors','sensors']#,'sensors','sensors']#,'sensors','sensors']#['sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors','sensors']#,'areas','areas','areas','areas']#,'sensors','sensors','sensors','sensors'] #,'areas','areas','areas','areas']#,['times','areas','speeds','times','areas','speeds']

# constraints
'''
Ha - pixels
12 192

15 240

18 288

25 400

30 480

45 560

60 960

80 1280

90 1440

100 1600

110 1760

120 1920
'''
# constraints = [['noConstraint',0] this is how you set a no constraint
#constraints = [['detectArea',0],['detectArea',480],['detectArea',800]]#[['detectArea',50],['detectArea',100],['detectArea',150],['detectArea',250],['detectArea',350],['detectArea',400],['detectArea',480],['detectArea',450],['detectArea',600],['detectArea',700],['detectArea',800],['detectArea',900],['detectArea',1500],['detectArea',1750],['detectArea',2000]]#,['detectTime',5400,400],['detectTime',7200,260,],['detectTime',9000,200],['detectTime',10800,180]]#,['detectTime',3600],['detectTime',2*3600],['detectTime',2*3600],['detectTime',2*3600]]
constraints = [['noConstraint',0],['nSensors',30]]#,['detectArea',200],['detectArea',300],['detectArea',400],['detectArea',1000],['detectArea',10000],['detectArea',321*321]]#,['detectArea',1440],['detectArea',1760],['detectArea',2080]]#,['detectArea',960],['detectArea',1280],['detectArea',1440],['detectArea',1600]]#,['detectArea',400],['detectArea',480],['detectArea',450],['detectArea',600],['detectArea',700],['detectArea',800],['detectArea',900],['detectArea',1500],['detectArea',1750],['detectArea',2000]]#,['detectTime',5400,400],['detectTime',7200,260,],['detectTime',9000,200],['detectTime',10800,180]]#,['detectTime',3600],['detectTime',2*3600],['detectTime',2*3600],['detectTime',2*3600]]
nTerminations = [400000,400000]# ,20000,20000,20000,20000,20000,20000,20000]#,400000,400000]#,400000,400000]#[200000,200000,200000,200000,200000,200000,200000,200000,200000,200000,200000,200000,200000,200000,200000]

npops = [150,300]# ,150,150,150,150,150,150,150]#,150,150]#,150,150] #[200,200,200,200,150,150,150,150,150,150,100,100,100,100,100]

# Optimization loop

# Paramters are being defined as lists. The loop is performed over the zipped lists
# Choose optimization by times or burned areas
    
for objective1,objective2,constraint,nTermination,npopp,constraint in zip(objectives1,objectives2,constraints,nTerminations,npops,constraints):

    # Adequating data to time constraint

    # Uncomment if fires/bAreas file is missing to recompute it
    firesAtSensors = [] # np.zeros((8,nig,ns))
    bAreas = [] # np.zero s((8,nig,ns))
    speedAtSensors = []
    nigs = []
    keep_igs = []
    not_keep_igs = []
    for i in range(8):
        #print("Computing simulated arrival times data")
        #fires = loadingFires(simDataPaths[i],validIgnitions,nx,ny)
        # Limiting fire arrival times at possible sensor locations
        #firesAtSensors[i,:,:] = fires[:,validSensors]
        #np.save(f'firesAtSensors_{i}',firesAtSensors[i,:,:],allow_pickle=True) # For this case study this file is about 3.7 Gb of data.
        print("Loading simulated arrival times data")
        firesData = np.load(f'firesAtSensors_{i}.npy',allow_pickle=True)

        #print("Computing burned areas at sensor locations")
        #bAreas[i,:,:] = burnedAtSensorLocation(fires,validIgnitions,validSensors,nx,ny)
        #burnedArea2file(fires,i,constraint)
        
        #np.save(f'bAreas_{i}',bAreas[i,:,:],allow_pickle=True)
        print("Loading burned areas at sensor locations")
        bAreasData = np.load(f'bAreas_{i}.npy',allow_pickle=True)
    
        print("Filtering data")
        keep_ig = []
        not_keep_ig = []
        # The criteria for ignoring a wildfire is that if wildfire approaches a sensor in more than 1 hour, it is slow enough to be detected by other means
        if constraint[0] == 'detectTime':
            for ii in range(nig):
                try:
                    nonZeroTimes = np.amin(firesData[ii,firesData[ii,:]>0]) # Non zero times
                    # If fire is faster, then it is important to keep it in the analysis
                    if nonZeroTimes <= constraint[1]:
                        keep_ig.append(ii)
                    else:
                        not_keep_ig.append(ii)
                except:
                    # If nonZeroTimes is null value, then it means the fire did not even spread to a possible sensor location, therefore ignoring this wildfire.
                    # Include it in not "keep"
                    not_keep_ig.append(ii)

        elif constraint[0] == 'detectArea':
            for ii in range(nig):
                try:
                    nonZeroAreas = np.amin(bAreasData[ii,bAreasData[ii,:]>0]) # Non zero areas
                    # If fire is faster, then it is important to keep it in the analysis
                    if nonZeroAreas <= constraint[1]:
                        keep_ig.append(ii)
                    else:
                        not_keep_ig.append(ii)
                except:
                    # If nonZeroTimes is null value, then it means the fire did not even spread to a possible sensor location, therefore ignoring this wildfire.
                    # Include it in not "keep"
                    not_keep_ig.append(ii)
        else:
            keep_ig = list(np.arange(nig,dtype=int))

        # New number of ignitions
        nigs.append(len(keep_ig))
        keep_igs.append(keep_ig)
        not_keep_igs.append(not_keep_ig)

        # Finally keeping the values as considered
        # Now we do not omit the sensors that do not satisfy the time constraint, we just  not which ones are and then inside optimizer do such operation
        #firesData = firesData[keep_ig,:]
        #bAreasData = bAreasData[keep_ig,:]

        # Computing average spread speed at detection
        #speedData = -bAreasData/firesData # Negative because intention will be to maximize speed at detection
        #speedData = -1./firesData # Negative because intention will be to maximize speed at detection
        #speedData[speedData==-np.inf] = penalizations[2]
        #speedData = np.nan_to_num(speedData,penalizations[2]) # Substitute Nan values by speed penalization.

        firesAtSensors.append(firesData)
        bAreas.append(bAreasData)
        #speedAtSensors.append(speedData)

        # Preparing data for optimization. Since data is sparse, during optimization the computation of objective functions may find as minimum null values, which obscure the optimization process.
        # This issue is solved by penalizing those pixels without wildfire presence with maximal values.
        firesAtSensors[-1][np.where(firesAtSensors[-1]==0.)] = penalizations[0] # Arrival times are extrapolated to the maximum simulated value (5 hours in this case, which is expressed in seconds, the units in which arrival times are presented)
        # Penalization areas is different, as each fire has different area. 
        # Area penalization is the largest area per fire.
        maxAreas = np.loadtxt(f'maxAreas_{i}')
        for ji in range(len(bAreas[-1])):
            bAreas[-1][ji,np.where(bAreas[-1][ji,:]==0.)] = maxAreas[ji]    # Burned areas are extrapolated to the maximum burnt area of the whole wildfire dataset dataset
        # Speed at sensors is already penalized 
    print(nigs)

    # It is important for analysis purposes 
    with open('keptFires', 'w') as file:
        for list_igs in keep_igs:
            file.write(' '.join(map(str, list_igs)) + '\n')

    # Save the lengths_list to another text file
    with open('nigs', 'w') as file:
        file.write(' '.join(map(str, nigs)) + '\n')

    # Sorting
    indexes = []
    for i in range(8):
        # Fast optimization requires to pre-sort the different arrival times / burned areas from the different wildfires detected by each sensor. Wildfires do not spreading over a given sensor have their corresponding sensor location pixel penalized and do not matter and their sorting has no role.
        indexes.append(sortingComputation(nig,ns,firesAtSensors[i])) # Now we are not omitting any fire
        
    weights = frequencies

    # Population values for optimization
    npop = npopp #constraint[1]
    nOffsprings = 20 # Number of offsprings during evolutionary generation

    # Instantiating Problem classes
    problem = so.optimalSensorLocWeigthed(keep_igs,not_keep_igs,ns,firesAtSensors,bAreas,indexes,npop,nOffsprings,objective1,objective2,weights,constraint,penalizations,so.numbaLoop)

    # Finally, computation of the pareto front solutions

    nTermination = nTermination
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
                callback=MyCallback(),
                save_history=False,
                seed = 1,
                verbose=True)

    # Creat animation
    #ma.animCase1(res.algorithm.callback,npop, ns, nig, validIgnitions, indexes, firesAtSensors, bAreas, penalizations,objective1,objective2,constraint,nTermination)
    # Save the object to a file using pickle
    with open(f'pickle_{objective1}_{objective2}_c_{constraint[0]}_v_{constraint[1]}_{nTermination}_npop_{npop}.pkl', 'wb') as file:
        pickle.dump(res.algorithm.callback, file)

    try:

        # Next statement gives problems
        #n_iters = res.algorithm.callback.data['n_iters']

        print("Best solution found: %s" % res.X.astype(int))
        print("Function value: %s" % res.F)
        print("Constraint violation: %s" % res.CV)

        Xt = res.X
        Ft = res.F
        try:
            np.savetxt(f'x_opt_{objective1}_{objective2}_c_{constraint[0]}_v_{constraint[1]}_{nTermination}_npop_{npop}',Xt)
            np.savetxt(f'Ft_opt_{objective1}_{objective2}_c_{constraint[0]}_v_{constraint[1]}_{nTermination}_npop_{npop}',Ft)
        except:
            np.savetxt(f'x_opt_{n_iters}',Xt)
            np.savetxt(f'Ft_opt_{n_iters}',Ft)
        # calculate a hash to show that all executions end with the same result
        print("hash", res.F.sum())
    except:
        print("No pareto was found")
   
