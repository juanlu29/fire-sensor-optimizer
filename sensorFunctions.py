import numpy as np
import os
import sys


def detectIgnitionsFaster(x, n_activated_sensors, ns, nig, ignitions, sensors, sorting, fires, bAreas, penalization):
    '''
    This function provides the ignitions detected for a given sensor configuration and set of fire simulations

    penalization is exclusively related to arrival times

    This function is optimized to analyze fast the solution results
    Essentially, restricts the size of the matrix according to the locations where sensors are placed. Information is destroyed,
    and it is that now the specific sensors positions cannot be inferred by the retrieval of non-zero rows

    This function does not control which fires are detected or not by certain constraints. It just computes the magnitudes detected by sensors
    '''
    # 3 dimensions.
    # First for number of ignitions
    # Second for the total minimum time registered
    # third for the areas registered

    if n_activated_sensors != len(sensors):
        print("Inconsistency in detectIgnitionsFaster function")
        print("Make sure that the sensors input to the function are the present-sensors in the configuration")
        sys.exit()
    try:
        fires_by_sensors = fires[x == 1].reshape((nig, n_activated_sensors))
        sorted_dx = np.argmin(fires_by_sensors, axis=1)
        sorting = sensors[sorted_dx]  # For the present sensors, indexing by sorted_dx retrieves those that detect each ignition accordingly. It is necessary to filter out later the fires detected at penalization

        ignitionsPerSensor = np.zeros((4, ns, nig))
        minimumTimes = np.zeros(nig, dtype=int)

        minimumTimes = fires[np.arange(nig, dtype=int), sorting]  # Just keep updating this value
        nonZeroTimes = np.where(minimumTimes != penalization)[0]  # When a fire is not detected, the sensor is associated to a penalization. This filters out penalized ignitions

        ignitionsPerSensor[0, sorting[nonZeroTimes], nonZeroTimes] = ignitions[nonZeroTimes]  # Just keep updating this value
        ignitionsPerSensor[1, sorting[nonZeroTimes], nonZeroTimes] = minimumTimes[nonZeroTimes]  # Just keep updating this value
        ignitionsPerSensor[2, sorting[nonZeroTimes], nonZeroTimes] = bAreas[nonZeroTimes, sorting[nonZeroTimes]]  # Just keep updating this value
        v = bAreas[nonZeroTimes, sorting[nonZeroTimes]] / minimumTimes[nonZeroTimes]
        ignitionsPerSensor[3, sorting[nonZeroTimes], nonZeroTimes] = v  # Spread speed. Just keep updating this value
    except:
        print("Error in detectIgnitionFaster function")
        print(n_activated_sensors, ns, nig)
        sys.exit()

    
    return ignitionsPerSensor


def detectIgnitions(x,ignitions,sorting,fires,bAreas,penalization):
    '''
    This function provides the ignitions detected for a given sensor configuration and set of fire simulations

    penalization is exclusively related to arrival times
    '''
    # 3 dimensions.
    # First for number of ignitions
    # Second for the total minimum time registered
    # third for the areas registered
    ignitionsPerSensor = np.zeros((3,len(x),len(ignitions)),dtype=int)
    minimumTimes = np.zeros(len(ignitions),dtype=int)
    # Each ignition can be sensed by one sensor at most. Therefore running through ignitions is enough
    for i in range(len(ignitions)):
        s = 0
        while x[sorting[i,s]] < 1.:
            s += 1
            if s == len(ignitions): # Already surpass number of sensors
                # It did not found any sensor to sense the fire
                s-=1
                break
        minimumTimes[i] = fires[i,sorting[i,s]] # Just keep updating this value
        ignitionsPerSensor[0,sorting[i,s],i] = ignitions[i] # Just keep updating this value
        ignitionsPerSensor[1,sorting[i,s],i] = minimumTimes[i] # Just keep updating this value
        ignitionsPerSensor[2,sorting[i,s],i] = bAreas[i,sorting[i,s]] # Just keep updating this value
        #ignitionsPerSensor[3,sorting[i,s],i] = np.nan_to_num(bAreas[i,sorting[i,s]]/minimumTimes[i],0) # Spread speed. Just keep updating this value
        if minimumTimes[i] == penalization: # Fire was not detected by sensor actually
            ignitionsPerSensor[0,sorting[i,s],i] = 0
            ignitionsPerSensor[1,sorting[i,s],i] = 0
            ignitionsPerSensor[2,sorting[i,s],i] = 0
            ignitionsPerSensor[3,sorting[i,s],i] = 0
            #ignitionsPerSensor[3,sorting[i,s],i] = 0
    
    return ignitionsPerSensor



def timesMapping(data,solution,validIgnitions,nx,ny,sliding):
    '''
    Given a solution matrix of the form sensor indexes in rows and ignition indexes in the columns, it produces a matrix
    that can be visualized as the spatial fragmentation of ignitions detected by the different sensors. 

    The segmentations are contoured to indicate an specific sensor covering area. Each ignition is coloured to indicate the arrival time when it was detected.

    data, computed using the function detectIgnitions(), has in the first coordinate when equals zero the ignition indexes, and in the second value the arrival times at detection

    '''

    # dataIgnitions
    dataIgnitions = data[0,:,:]
    minimumTimes = data[1,:,:]

    print("Defining basic spatial location arrangement")
    # First defining the overgrid that represents each sensor/ignition location. Ignitions are a regular subsampling of the domain
    # Therefore, each pixel location must be surrounded by different equivalent pixels.
    # Defining the matrix
    fullMapping = np.zeros((ny,nx),dtype=int)

    # Each sensor/ignition location is separated by an amount of pixels. The sliding squared that will overlap areas for a given location is represented
    slidingSquared = np.zeros((ny,nx),dtype=int)
    slidingSquared[0:sliding,0:sliding] = 1

    # Finally mapping each pixel-surface element with the corresponding index location
    for susceptible in validIgnitions:
        A = np.roll(slidingSquared,(susceptible%nx,(susceptible/ny).astype(int)),axis=(1,0))
        fullMapping[np.where(A==1)[0],np.where(A==1)[1]] = susceptible

    print("---")
    print("Computing ignitions that have not been sensed")

    sensed = np.unique(dataIgnitions[dataIgnitions!=0])
    notSensed = []
    #  Computing non sensed
    for ignS in validIgnitions:
        sensedBool = False
        for ignSS in sensed:
            if ignS == ignSS:
                # If the script flow goes here, it means the ignition has been sensed.
                sensedBool = True
                break
        if sensedBool == False:
            notSensed.append(ignS)

    notSensed = np.asarray(notSensed)

    # Associating a -1 value to identify notSensed elements
    for nSt in notSensed:
        fullMapping[fullMapping==nSt] = -1

    print("---")
    print("Now identifying detected ignition sectors meanwhile computing contours in parallel")

    c = 1
    cs = []
    sensorList = np.arange(len(solution),dtype=int)
    sensorList = sensorList[np.where(solution!=0)]
    fullMappingCopy = fullMapping.copy() # A copy is made to avoid over-writing in fullMapping values during the loop
    fullMappingCopyCopy = fullMapping.copy()
    for senIndex in sensorList:
        sensByIgns = dataIgnitions[senIndex,:]
        for ix, ignSensed in enumerate(sensByIgns[sensByIgns!=0]):
            if len(sensByIgns[sensByIgns!=0])>0:
                fullMapping[fullMappingCopy == ignSensed] = minimumTimes[senIndex,minimumTimes[senIndex]!=0][ix]/3600. # Assigning this value, there is a correspondence between the color and the number of detected wildfires
                fullMappingCopyCopy[fullMappingCopy == ignSensed] = c # Use for contouring. Otherwise, if two patches detect the same amount of fires, the contouring does not work adequately
        c += 1
        cs.append(len(sensByIgns[sensByIgns!=0]))

    # For python-index nature
    c -= 1

    # Defining contours as how many c sensors that identified at least one ignition
    contours = np.zeros((c,fullMapping.shape[0],fullMapping.shape[1]),dtype=int)
    
    for ci in range(c):

        AAA = contours[ci,:,:].copy()
        AAA[fullMappingCopyCopy == ci+1] = 1
        #AAA[fullMapping == cs[ci]] = 1
        contours[ci,:,:] = AAA

        # Deriving countours
        countoursPlusUp = np.roll(contours[ci,:,:],2,axis=1)
        countoursPlusRight =  np.roll(contours[ci,:,:],2,axis=0)

        A = contours[ci,:,:]-countoursPlusUp
        A[A==1] = -2
        A[A==-1] = -2
        A[A!=-2] = +0

        B = contours[ci,:,:]-countoursPlusRight
        A[B==1] = -2
        A[B==-1] = -2
        A[A!=-2] = 0

        contours[ci,:,:] = A


    
    print("---")
    print("Adding contours to image")

    contourMapping = fullMapping.copy()
    
    for ci in range(c):
        contourMapping[contours[ci,:,:]==-2] = -2

    return fullMapping, contours, contourMapping


def ignitionMapping(dataIgnitions,solution,validIgnitions,nx,ny,sliding):
    '''
    Given a solution matrix of the form sensor indexes in rows and ignition indexes in the columns, it produces a matrix
    that can be visualized as the spatial fragmentation of ignitions detected by the different sensors

    dataIgnitions is such data matrix. It is computed using the function detectIgnitions()

    '''

    print("Defining basic spatial location arrangement")
    # First defining the overgrid that represents each sensor/ignition location. Ignitions are a regular subsampling of the domain
    # Therefore, each pixel location must be surrounded by different equivalent pixels.
    # Defining the matrix
    fullMapping = np.zeros((ny,nx),dtype=int)

    # Each sensor/ignition location is separated by an amount of pixels. The sliding squared that will overlap areas for a given location is represented
    slidingSquared = np.zeros((ny,nx),dtype=int)
    slidingSquared[0:sliding,0:sliding] = 1

    # Finally mapping each pixel-surface element with the corresponding index location
    for susceptible in validIgnitions:
        A = np.roll(slidingSquared,(susceptible%nx,(susceptible/ny).astype(int)),axis=(1,0))
        fullMapping[np.where(A==1)[0],np.where(A==1)[1]] = susceptible

    print("---")
    print("Computing ignitions that have not been sensed")

    sensed = np.unique(dataIgnitions[dataIgnitions!=0])
    notSensed = []
    #  Computing non sensed
    for ignS in validIgnitions:
        sensedBool = False
        for ignSS in sensed:
            if ignS == ignSS:
                # If the script flow goes here, it means the ignition has been sensed.
                sensedBool = True
                break
        if sensedBool == False:
            notSensed.append(ignS)

    notSensed = np.asarray(notSensed)

    # Associating a -1 value to identify notSensed elements
    for nSt in notSensed:
        fullMapping[fullMapping==nSt] = -1

    print("---")
    print("Now identifying detected ignition sectors meanwhile computing contours in parallel")

    c = 1
    cs = []
    sensorList = np.arange(len(solution),dtype=int)
    sensorList = sensorList[np.where(solution!=0)]
    fullMappingCopy = fullMapping.copy() # A copy is made to avoid over-writing in fullMapping values during the loop
    fullMappingCopyCopy = fullMapping.copy()
    for senIndex in sensorList:
        sensByIgns = dataIgnitions[senIndex,:]
        for ignSensed in sensByIgns[sensByIgns!=0]:
            if len(sensByIgns[sensByIgns!=0])>0:
                fullMapping[fullMappingCopy == ignSensed] = len(sensByIgns[sensByIgns!=0]) # Assigning this value, there is a correspondence between the color and the number of detected wildfires
                fullMappingCopyCopy[fullMappingCopy == ignSensed] = c # Use for contouring. Otherwise, if two patches detect the same amount of fires, the contouring does not work adequately
        c += 1
        cs.append(len(sensByIgns[sensByIgns!=0]))

    # For python-index nature
    c -= 1

    # Defining contours as how many c sensors that identified at least one ignition
    contours = np.zeros((c,fullMapping.shape[0],fullMapping.shape[1]),dtype=int)
    
    for ci in range(c):

        AAA = contours[ci,:,:].copy()
        AAA[fullMappingCopyCopy == ci+1] = 1
        #AAA[fullMapping == cs[ci]] = 1
        contours[ci,:,:] = AAA

        # Deriving countours
        countoursPlusUp = np.roll(contours[ci,:,:],2,axis=1)
        countoursPlusRight =  np.roll(contours[ci,:,:],2,axis=0)

        A = contours[ci,:,:]-countoursPlusUp
        A[A==1] = -2
        A[A==-1] = -2
        A[A!=-2] = +0

        B = contours[ci,:,:]-countoursPlusRight
        A[B==1] = -2
        A[B==-1] = -2
        A[A!=-2] = 0

        contours[ci,:,:] = A


    
    print("---")
    print("Adding contours to image")

    contourMapping = fullMapping.copy()
    
    for ci in range(c):
        contourMapping[contours[ci,:,:]==-2] = -2

    return fullMapping, contours, contourMapping


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


def burntCurved(fire):
    '''
    Given a fire instance, compute the burned curve altogether the time list
    ''' 
    times = np.sort(np.unique(fire[fire>0.]))
    areas = np.zeros(len(times))
    area = 0.
    for i,time in enumerate(times):
        area += len(fire[fire==time])
        areas[i] = area

    return areas, times


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