import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from math import *
import time
from catenarylib import findCatenary
from planelib import *
from plotlib import plot,plot2D

def cluster(data_numpy_array):#clustering function to find wires by agglomeration return data grouped by wire detected
    clustering = AgglomerativeClustering(n_clusters=None,linkage='single',distance_threshold=0.75).fit(data_numpy_array)#link points/agglomerations when the minimum distance between the points/aglomerations is inferior to distance_threshold with all the points in data_numpy_array
    #reorganizing datas into  [[points from wire 1],...,[points from wire n]]
    data_by_wire=[[] for _ in range(clustering.n_clusters_)]
    for i in range(len(clustering.labels_)):
        data_by_wire[clustering.labels_[i]].append(data_numpy_array[i])
    return data_by_wire

def main(): 
    df = pd.read_parquet('../data/lidar_cable_points_easy.parquet') #extracting data into df
    data_numpy_array = df.to_numpy()#create a numpy array from data frame

    data_by_wire = cluster(data_numpy_array)#clustering
    #plot(data_by_wire,len(data_by_wire))
    print("there are "+str(len(data_by_wire))+" wires detected by clustering") 

    planes=[findPlane(array) for array in data_by_wire]#finding planes
    bests_c_x0_y0=[]
    t0=time.time()
    for plane in planes:
        bests_c_x0_y0.append(findCatenary(plane[3]))
    end=time.time()-t0

    print("RESULTS : ")
    print("Time to find catenary functions in 2D: "+str(end))
    print("For catenary equation, the equation becomes :")
    print("  y1*x + y2*y + y3*z = y0 + c * [cosh((x1*x + x2*y + x3*z - x0)/c)-1]")
    for i in range(len(bests_c_x0_y0)):
        print("For wire "+str(i)+" :")
        print("  c = "+str(bests_c_x0_y0[i][0]))
        print("  x0 = "+str(bests_c_x0_y0[i][1]))
        print("  y0 = "+str(bests_c_x0_y0[i][2]))
        x=planes[i][4].inverse_transform(np.array([1,0]))
        y=planes[i][4].inverse_transform(np.array([0,1]))
        print("  x1 = "+str(x[0])+"; x2 = "+str(x[1])+";x3 = "+str(x[2]))
        print("  y1 = "+str(y[0])+"; y2 = "+str(y[1])+";y3 = "+str(y[2]))
        plot2D(planes[i][3],True,bests_c_x0_y0[i][1],bests_c_x0_y0[i][2],bests_c_x0_y0[i][0])

if __name__ == '__main__' :
    main()
