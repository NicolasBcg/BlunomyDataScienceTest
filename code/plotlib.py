import matplotlib.pyplot as plt
import numpy as np
from math import *
colors=['#808080', '#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#FF00FF', '#C0C0C0', '#FFA500', '#191970', '#17becf', '#FF69B4', '#8B008B', '#6B8E23', '#00BFFF'] #a bunch of color to display different labels

def plot(array=[],number_of_labels=0,planes=[]):#a function to visualise 3D area. no number_of_label if there are no labels
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if number_of_labels==0:#if we want to take labels into account
        for point in array:
            ax.scatter(point[0], point[1], point[2])
    else: #if labels are taken into account
        if number_of_labels > len(colors):
            for _ in range(len(colors),number_of_labels+1):
                colors.append('#000000')
        for label_number in range(number_of_labels):
            for point in array[label_number]:
                ax.scatter(point[0], point[1], point[2],c=colors[label_number])
    for plane in planes:#if we want to display planes
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        X,Y = np.meshgrid(np.arange(xlim[0], xlim[1]),
                        np.arange(ylim[0], ylim[1]))
        Z = np.zeros(X.shape)
        for r in range(X.shape[0]):
            for c in range(X.shape[1]):
                Z[r,c] = plane[0] * X[r,c] + plane[1] * Y[r,c] + plane[2]
        ax.plot_wireframe(X,Y,Z, color='k')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def plot2D(array=[],catenary=False,x0=0,y0=0,c=1):#a function to visualise 2D data array.
    for point in array:
        plt.scatter(point[0], point[1],c=colors[0])
    if catenary:
        x = [-25+i/2 for i in range(100)]
        y = [y0+c*cosh((xi-x0)/c)-c for xi in x]
        plt.plot(x, y)
    plt.show()