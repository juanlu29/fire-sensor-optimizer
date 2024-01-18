Pymoo Problem class (sensorProblemClass.py) file to optimize wireless sensor network locations for autonomous wildfire detectors using wildfire simulations from https://www.sciencedirect.com/science/article/pii/S1470160X23014486.

Along with the class, a Python script optimizingScript.py executes the optimization on a test case.

The figure sensorLocations_ignitionLocations.png represents the ignition locations for wildfire simulations (x) and the possible sensor locations (o). The domain use to test this methodology corresponds to the case study computed for the previous paper.

![alt text](https://github.com/juanlu29/fire-sensor-optimizer/blob/main/sensorLocations_ignitionLocations.png?raw=true)

Each wildfire is simulated 5 hours. The optimization routine selects sensor configurations minimizing arrival time or wildfire burned areas at sensor detection.

To run this example, a series of Python libraries are needed. The environment.yml replicates the conda environement in which this optimization is performed. 

If everything goes fine, the parete front should be displayed

![alt text](https://github.com/juanlu29/fire-sensor-optimizer/blob/main/testParetoFront.png?raw=true)
