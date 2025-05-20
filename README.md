This repository contains the code to reproduce the Pareto front from the study "Simulation-driven Optimization for the Localization of Wireless Sensor Networks for Early Wildfire Detection" submitted to the journal "Environmental Modelling & Software". It makes use of Python code and the optimization framework Pymoo (J. Blank and K. Deb, 2020). It contains the Pymoo Problem class (sensorProblemClass.py), defining the objectives to optimize wireless sensor network (WSN) locations for early wildfire detection. The wildfire observable chosen to trigger the sensors is the fire perimeter, and the wildfire attribute considered as a performance estimator in the research study is the burnt area. Additionally, this implementation allows for minimizing the arrival times of the fire perimeters to the sensor locations. The wildfire simulator used to generate the wildfire dynamic data for this study is taken from (Gómez-González et al. 2024).

Along with the class, a Python script optimizingScript.py instantiates the Pymoo Problem class and runs the optimization on a test case. The figure sensorLocations_ignitionLocations.png represents the ignition locations for wildfire simulations (blue squares) and the possible sensor locations (orange squares). Possible sensor locations extend up to the domain's borders while the ignition grid is offset from the boundaries. 

![alt text](https://github.com/juanlu29/fire-sensor-optimizer/blob/main/grid_ignitions_sensors_zoomed.png?raw=true)

The computation of fire behaviour follows the methodology of (Gómez-González et al. 2024). Each ignition corresponds to a wildfire simulated for 5 hours. As stated above, for optimization, one can search optimal sensor configurations based on minimizing fire arrival time or wildfire-burned areas at sensor detection. 

The paper describes a methodology to select high-risk fire-weather scenarios. As a result, simulations are run for a wind speed with cardinal wind direction. These simulations are uploades as zipped files Sims_p90_X.zip, where X stands for N (North), S(South), SW (South-west), and so on. Then, the maximum burnt area per wildfire are stored in the max_ba.zip file. Only a series of tabulated data is necessary to run the optimization. This information is zipped in the caseStudy.zip file.

To run this example, a series of Python libraries are needed. The environment.yml files replicate the conda environment in which this optimization is performed. 

As a result, one derives a series of solutions whose performance defines the Pareto-front of the problem. The following figure represents the Pareto front, which results from minimizing burnt areas, while the inset reproduces the corresponding average detection times. As a performance estimation, detection times can be computed from an arbitrary WSN, not implying optimization.

![alt text](https://github.com/juanlu29/fire-sensor-optimizer/blob/main/paretoPerformance.png?raw=true)

References,

(Gómez-González et al. 2024); Juan Luis Gómez-González, Alexis Cantizano, Raquel Caro-Carretero, Mario Castro, Leveraging national forestry data repositories to advocate wildfire modelling towards simulation-driven risk assessment, Ecological Indicators, Volume 158, 2024, 111306, ISSN 1470-160X, doi=https://doi.org/10.1016/j.ecolind.2023.111306. url=https://www.sciencedirect.com/science/article/pii/S1470160X23014486.

 (J. Blank and K. Deb, 2020); J. Blank and K. Deb, pymoo: Multi-Objective Optimization in Python, in IEEE Access, vol. 8, pp. 89497-89509, 2020, doi: 10.1109/ACCESS.2020.2990567


If you find this data and approach useful, please cite the following article

@article{GOMEZGONZALEZ2025113509,
title = {Simulated fire observables as indicators for optimizing wireless sensor networks in wildfire risk monitoring},
journal = {Ecological Indicators},
volume = {175},
pages = {113509},
year = {2025},
issn = {1470-160X},
doi = {https://doi.org/10.1016/j.ecolind.2025.113509},
url = {https://www.sciencedirect.com/science/article/pii/S1470160X2500439X},
author = {Juan Luis Gómez-González and Effie Marcoulaki and Alexis Cantizano and Myrto Konstantinidou and Raquel Caro and Mario Castro},
keywords = {Wildfire simulation, Optimization, Early wildfire detection system, Wireless sensor network, NSGA-II},
}
