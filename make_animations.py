# Importing problem class
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from matplotlib.animation import FuncAnimation
import sensorProblemClass as so
import sensorFunctions as sf
from pymoo.core.callback import Callback
import sys


def animCase1(resCallback, npop, ns, nig, validIgnitions, indexes, firesAtSensors, bAreas, penalizations, objective1,
              objective2, constraint, nTermination):
    sensors = np.arange(ns, dtype=int)

    val = resCallback.data["best"]
    # for vv in range(len(val)):
    #    val[vv][:,0] = val[vv][:,0]/3600.
    n_iters = resCallback.data["n_iter"]
    sols = resCallback.data["sols"]

    # Reading max areas
    sumAreas = 0
    for i in range(8):
        maxAreas_ = np.loadtxt(f'maxAreas_{i}')
        sumAreas += np.sum(maxAreas_)

    # Initialization function: plot the background of each frame
    def init():
        scatter.set_offsets(val[0])
        # It is necessary to creat maximum area data
        maxAreas = np.zeros((npop, 2))
        for aa, xx in enumerate(sols[0]):
            x = xx.astype(int)
            n_activated_sensors = np.sum(x)
            present_sensors = sensors[x == 1]
            # For each wind
            for jj in range(8):
                # Expanding solution
                x_expanded = np.zeros((nig, ns), dtype=int)
                x_expanded[:, :] = x
                data = sf.detectIgnitionsFaster(x_expanded, n_activated_sensors, ns, nig, validIgnitions,
                                                present_sensors, indexes[jj], firesAtSensors[jj], bAreas[jj],
                                                penalizations[0])
                # if  np.amax(data[2,:,:]) > maxAreas[aa,1]:
                #    maxAreas[aa,1] =  np.amax(data[2,:,:])
                maxAreas[aa, 1] += np.sum(data[2, data[2, :, :] > 0])
            maxAreas[aa, 0] = val[0][aa, 0]
            # Remove two next statements, it is a very cruce visualization
            maxAreas[:,0] = val[0][:,0]
            maxAreas[:,1] = maxAreas[aa,1]
            break
        scatter2.set_offsets(maxAreas)
        return scatter,

    # Animation function. This is called sequentially
    def animate(i):
        ax.set_title(f'# gen {n_iters[int(i)]}')
        scatter.set_offsets(val[int(i)])
        # It is necessary to creat maximum area data
        maxAreas = np.zeros((npop, 2))
        for aa, xx in enumerate(sols[i]):
            x = xx.astype(int)
            n_activated_sensors = np.sum(x)
            present_sensors = sensors[x == 1]

            # For each wind
            for jj in range(8):
                # Expanding solution
                x_expanded = np.zeros((nig, ns), dtype=int)
                x_expanded[:, :] = x
                data = sf.detectIgnitionsFaster(x_expanded, n_activated_sensors, ns, nig, validIgnitions,
                                                present_sensors, indexes[jj], firesAtSensors[jj], bAreas[jj],
                                                penalizations[0])
                # if  np.amax(data[2,:,:]) > maxAreas[aa,1]:
                #    maxAreas[aa,1] =  np.amax(data[2,:,:])
                maxAreas[aa, 1] += np.sum(data[2, data[2, :, :] > 0])
            maxAreas[aa, 0] = val[int(i)][aa, 0]
            # Remove two next statements, it is a very cruce visualization
            maxAreas[:,0] = val[i][:,0]
            maxAreas[:,1] = maxAreas[aa,1]
            break
        scatter2.set_offsets(maxAreas)
        print(f'# gen frame {i} generated')
        return scatter,

    # First set up the figure, the axis, and the plot element we want to animate
    figure, ax = plt.subplots()

    # Setting final frame lims. Final pareto is expected to cover maximum area of exploration
    ax.set_xlim([np.amin(val[-1][:, 0]), np.amax(val[-1][:, 0])])
    ax.set_ylim([np.amin(val[-1][:, 1]), np.amax(val[-1][:, 1])])
    ax.set_ylabel('obj: # sensors')
    ax.set_xlabel('obj: minimize time / seconds')
    scatter = ax.scatter(val[0][:, 0], val[0][:, 1], color='blue', label='pareto points')
    xmin = 1e30
    xmax = 0
    ymin = 1e30
    ymax = 0
    for va in val:
        if np.amin(va[:, 0]) < xmin:
            xmin = np.amin(va[:, 0])
        if np.max(va[:, 0]) > xmax:
            xmax = np.max(va[:, 0])
        if np.amin(va[:, 1]) < ymin:
            ymin = np.amin(va[:, 1])
        if np.max(va[:, 1]) > ymax:
            ymax = np.max(va[:, 1])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    # Create a second axis sharing the same x-axis
    ax2 = ax.twinx()
    ax2.set_ylabel('maximum area / pixels')

    # Plot the second set of data on the second axis (customize this part as needed)
    # It is necessary to creat maximum area data
    maxAreas = np.zeros((npop, 2))
    for aa, xx in enumerate(sols[0]):
        x = xx.astype(int)
        n_activated_sensors = np.sum(x)
        present_sensors = sensors[x == 1]
        # For each wind
        for jj in range(8):
            # Expanding solution
            x_expanded = np.zeros((nig, ns), dtype=int)
            x_expanded[:, :] = x
            data = sf.detectIgnitionsFaster(x_expanded, n_activated_sensors, ns, nig, validIgnitions, present_sensors,
                                            indexes[jj], firesAtSensors[jj], bAreas[jj], penalizations[0])
            # if  np.amax(data[2,:,:]) > maxAreas[aa,1]:
            #    maxAreas[aa,1] =  np.amax(data[2,:,:])
            maxAreas[aa, 1] += np.sum(data[2, data[2, :, :] > 0])
        maxAreas[aa, 0] = val[0][aa, 0]
        # Remove two next statements, it is a very cruce visualization
        maxAreas[:,0] = val[0][:,0]
        maxAreas[:,1] = maxAreas[aa,1]
        break

    # Just to define second axis limit
    maxAreas = np.zeros((npop, 2))
    for aa, xx in enumerate(sols[-1]):
        x = xx.astype(int)
        n_activated_sensors = np.sum(x)
        present_sensors = sensors[x == 1]
        # For each wind
        for jj in range(8):
            # Expanding solution
            x_expanded = np.zeros((nig, ns), dtype=int)
            x_expanded[:, :] = x
            data = sf.detectIgnitionsFaster(x_expanded, n_activated_sensors, ns, nig, validIgnitions, present_sensors,
                                            indexes[jj], firesAtSensors[jj], bAreas[jj], penalizations[0])
            # if  np.amax(data[2,:,:]) > maxAreas[aa,1]:
            #    maxAreas[aa,1] =  np.amax(data[2,:,:])
            maxAreas[aa, 1] += np.sum(data[2, data[2, :, :] > 0])
        maxAreas[aa, 0] = val[0][aa, 0]
        # Remove two next statements, it is a very cruce visualization
        maxAreas[:,0] = val[0][:,0]
        maxAreas[:,1] = maxAreas[aa,1]
        break

    y_limite = np.amax(maxAreas[:, 1])

    scatter2 = ax2.scatter(maxAreas[:, 0], maxAreas[:, 0], color='green',
                           label='sum detected areas')  # Example: Multiply the data for the second axis
    # ax2.set_ylim(0,constraint[1]*(1.20)) # +20% of the maximum constraint value
    ax2.set_ylim(0, y_limite * 1.20)
    ax2.axhline(y=constraint[1], color='r', linestyle='-', label='constraint')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Since plotting a single graph
    animation = FuncAnimation(
        figure, func=animate, frames=np.arange(0, len(val), 1), interval=500, init_func=init, repeat_delay=1000
    )
    # plt.show()

    animation.save(f'pareto_conv_{objective1}_{objective2}_c_{constraint[0]}_v_{constraint[1]}_{nTermination}.gif',
                   writer='PillowWriter', fps=4)
    plt.close()


def animCase2(res_list, npop, ns, nig, validIgnitions, indexes, firesAtSensors, bAreas, penalizations, objective1,
              objective2, constraints, nTermination):
    vals = [res.data["best"] for res in res_list]
    # for vvv in range(len(vals)):
    #    for vv in range(len(vals[vvv])):
    #        vals[vvv][vv][:, 0] = vals[vvv][vv][:, 0] / 3600.

    n_iterss = [res.data["n_iter"] for res in res_list]
    solss = [res.data["sols"] for res in res_list]

    # Initialization function: plot the background of each frame
    def init():
        [scatter.set_offsets(vals[vvv][0]) for vvv, scatter in enumerate(scatters1)]
        [scatter.set_offsets(np.asarray([vals[vvv][0][:, 0], np.sum(solss[vvv][0], axis=1)]).T) for vvv, scatter in
         enumerate(scatters2)]

        ax.legend(handles=legend_handles, labels=legend_labels)  # Update legend

        return scatters1, scatters2

    # Animation function. This is called sequentially
    def animate(i):
        ax.set_title(f'# gen {n_iterss[0][int(i)]}')

        # combined_offsets = np.concatenate([vals[vvv][int(i)] for vvv in range(len(vals))])
        # print(vals[0][-1][:,0],len(vals[0][-1][:,0]))
        # print(np.sum(solss[0][-1],axis=1),len(np.sum(solss[0][-1],axis=1)))
        print(np.asarray([vals[0][int(i)][:, 0], np.sum(solss[0][int(i)], axis=1)]).T)
        # combined_offsets_2 = np.concatenate([np.asarray([vals[vvv][int(i)][:,0],np.sum(solss[vvv][int(i)],axis=1)]).T for vvv in range(len(vals))])
        [scatter.set_offsets(vals[vvv][int(i)]) for vvv, scatter in enumerate(scatters1)]
        [scatter.set_offsets(np.asarray([vals[vvv][int(i)][:, 0], np.sum(solss[vvv][int(i)], axis=1)]).T) for
         vvv, scatter in enumerate(scatters2)]
        # scatter.set_offsets(combined_offsets)
        ax.legend(handles=legend_handles, labels=legend_labels)  # Update legend

        print(f'# gen frame {i} generated')
        return scatter1, scatters2

    # First set up the figure, the axis, and the plot element we want to animate
    figure, ax = plt.subplots()

    # Setting final frame lims. Final pareto is expected to cover maximum area of exploration

    xmin = 1e30
    xmax = 0
    ymin = 1e30
    ymax = 0

    colors = ['red', 'orange', 'green', 'blue', 'purple']  # You may have to change this according to the size of vals

    for val in vals:
        for va in val:
            if np.amin(va[:, 0]) < xmin:
                xmin = np.amin(va[:, 0])
            if np.amax(va[:, 0]) > xmax:
                xmax = np.amax(va[:, 0])
            if np.amin(va[:, 1]) < ymin:
                ymin = np.amin(va[:, 1])
            if np.amax(va[:, 1]) > ymax:
                ymax = np.amax(va[:, 1])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_ylabel('obj: minimize area / pixels')
    ax.set_xlabel('obj: minimize time / hours')

    # Create legend handles and labels
    legend_handles = []
    legend_labels = []

    scatters1 = []
    for vvv in range(len(vals)):
        # color = 'C{}'.format(vvv)  # Assign a different color for each dataset
        handle = Line2D([0], [0], marker='o', color=colors[vvv], markerfacecolor=colors[vvv], markersize=8,
                        label=f'# sensors {constraints[vvv][1]}')
        legend_handles.append(handle)
        legend_labels.append(f'# sensors {constraints[vvv][1]}')

        # Initialize scatter plot with the specified color
        scatter1 = ax.scatter(vals[vvv][0][:, 0], vals[vvv][0][:, 1], color=colors[vvv], marker='o')
        scatters1.append(scatter1)

    # Create a second axis sharing the same x-axis
    ax2 = ax.twinx()
    ax2.set_ylim(0, 200)
    ax2.set_ylabel('number sensors')
    scatters2 = []
    for vvv in range(len(vals)):
        scatter2 = ax2.scatter(vals[vvv][0][:, 0], np.sum(solss[vvv][0], axis=1), color=colors[vvv],
                               marker='x')  # Example: Multiply the data for the second axis
        scatters2.append(scatter2)

    # Initialize legend
    ax.legend(handles=legend_handles, labels=legend_labels, loc='upper left')
    # ax2.legend(loc='upper right')

    # Since plotting a single graph
    animation = FuncAnimation(
        figure, func=animate, frames=np.arange(0, len(vals[0]), 1), interval=500, init_func=init, repeat_delay=1000
    )
    # plt.show()
    animation.save(
        f'pareto_case_2_conv_{objective1}_{objective2}_c_{constraints[0][1]}_to_{constraints[-1][1]}_{nTermination}.gif',
        writer='PillowWriter', fps=4)
    plt.close()

