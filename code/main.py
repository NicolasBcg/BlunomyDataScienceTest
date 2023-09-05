import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from math import *
from scipy.optimize import fmin
import time
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

def plot2D(array,catenary=True,x0=0,y0=0,c=1):#a function to visualise 2D data array.
    for point in array:
        plt.scatter(point[0], point[1],c=colors[0])
    if catenary:
        x = [-25+i/2 for i in range(100)]
        y = [y0+c*cosh((xi-x0)/c)-c for xi in x]
        plt.plot(x, y)
    plt.show()

def cluster(data_numpy_array):#clustering function to find wires by agglomeration
    clustering = AgglomerativeClustering(n_clusters=None,linkage='single',distance_threshold=0.75).fit(data_numpy_array)#link points/agglomerations when the minimum distance between the points/aglomerations is inferior to distance_threshold with all the points in data_numpy_array
    #reorganizing datas into  [[points from wire 1],...,[points from wire n]]
    data_by_label=[[] for _ in range(clustering.n_clusters_)]
    for i in range(len(clustering.labels_)):
        data_by_label[clustering.labels_[i]].append(data_numpy_array[i])
    return data_by_label

def findPlane(array): #find the nearest plane to an array of points and the points from the wire projected in to this plane
    pca = PCA(n_components=2)
    pca.fit(array)
    points=pca.inverse_transform(np.array([[0,0],[1,0],[0,1]]))#getting 3 points on the plane in 3D base
    v1=[points[1][i]-points[0][i] for i in range(3)]
    v2=[points[2][i]-points[0][i] for i in range(3)]
    p=[v1[1]*v2[2]-v1[2]*v2[1],v1[2]*v2[0]-v1[0]*v2[2],v1[0]*v2[1]-v1[1]*v2[0]]#finding a vector which is orthogonal to the plane

    #calculating the solutions as ax + by + d = z
    a = p[0]/p[2]
    b = p[1]/p[2]
    d = (-p[0]*points[0][0] - p[1]*points[0][1] -  p[2]*points[0][2])/p[2]
    #print("%f x + %f y + %f = z" % (a, b, d))
    wire_projection=pca.transform(array)
    return a ,b ,d,wire_projection,pca

def distance_point_plane(point,plane):#gives the distance between a plane and a point
    return abs(plane[0]*point[0]+plane[1]*point[1]-point[2]+plane[2])/sqrt(pow(plane[0],2)+pow(plane[1],2)+1)

def catenary_dist_function(x,point,x0,y0,c):#gives the distance between a plane and a point from catenary
    return abs(sqrt(pow(point[0]-x,2)+ pow( point[1]-(y0+c*cosh((x-x0)/c)-c),2) ))

def minimum_distance_point_catanary(point,x0,y0,c):#gives the minimum distance between a plane and a point from catenary
    return fmin(catenary_dist_function, 0,args=(point,x0,y0,c),full_output=1,disp=0)

def error_on_catenary(x,array,x0,y0):#return the global error of the catenary function
    error=0
    for point in array:
        error+= minimum_distance_point_catanary(point,x0,y0,x)[1]
    return error

def find_x0_y0(array,spanReduction=1000):#function to find a suitable y0 !!! can be upgraded for more precision !!!
    xmin,xmax=0,0
    for point in array:#searching for xspan
        if point[0]>xmax:
            xmax=point[0]
        elif point[0]<xmin:
            xmin=point[0]
    span=xmax-xmin
    xspan_for_y=span/spanReduction
    ysum=0
    found=0
    for point in array:
        if point[0] > 0 - xspan_for_y  and point[0] < 0+xspan_for_y:
            ysum+=point[1]
            found+=1
    if found<len(array)/50:
        return find_x0_y0(array,spanReduction/2)
    elif found>len(array)/25:
        return find_x0_y0(array,spanReduction*1.5)
    else:
        return 0,ysum/found

def main(): 
     
    df = pd.read_parquet('../data/lidar_cable_points_easy.parquet') #extracting data into df
    data_numpy_array = df.to_numpy()#create a numpy array from data frame

    data_by_label = cluster(data_numpy_array)#clustering
    #plot(data_by_label,len(data_by_label))
    print("there are "+str(len(data_by_label))+" wires detected by clustering") 

    planes=[findPlane(array) for array in data_by_label]#finding planes
    bests_c_x0_y0=[]
    t0=time.time()
    for plane in planes:
        x0,y0=find_x0_y0(plane[3])
        # print(x0)
        # print(y0)
        bests_c_x0_y0.append([fmin(error_on_catenary, 1,args=(plane[3],x0,y0),full_output=1,disp=0)[0][0],x0,y0])
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

        plot2D(planes[i][3],True,10,bests_c_x0_y0[i][2],bests_c_x0_y0[i][0])

if __name__ == '__main__' :
    main()
