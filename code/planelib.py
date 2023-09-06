from sklearn.decomposition import PCA
from sklearn import linear_model
import numpy as np
from math import *
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

def sort(data_by_wire):
    sizes=[len(batch) for batch in data_by_wire]
    sizes.sort(reverse=True)
    return sizes

def mean_distance_batch_plane(batch,plane):#gives the distance between a plane and a point
    mean=0
    for point in batch:
        mean+=distance_point_plane(point,plane)
    return mean/len(batch)


def linkBatch(index,data_by_wire,linkage_threshold=0.25):#link batch based on the proximity of the points to the other's bach plane
    plane=findPlane(data_by_wire[index])
    #print(mean_distance_batch_plane(batch,plane))
    for i in range(len(data_by_wire)):
        if mean_distance_batch_plane(data_by_wire[i],plane)<linkage_threshold and i!=index:
            for point in data_by_wire[i]:
                data_by_wire[index].append(point)
            data_by_wire.pop(i)
            return data_by_wire,True
    return [],False


def agglomerateWithPlane(data_by_wire):#make plane based agglomeration
    sizes=sort(data_by_wire)
    for size in sizes:
        if size > 20:
            for i in range(len(data_by_wire)):
                if len(data_by_wire[i])==size:
                    res=linkBatch(i,data_by_wire)
                    if res[1] == True:
                        return agglomerateWithPlane(res[0])
    return data_by_wire


# all the same as plane agglomeration but with line drawn on plane x,y
def findLine(array):
    points=[[point[0],point[1]]for point in array]
    y=[point[1]for point in array]
    reg = linear_model.LinearRegression()
    reg.fit(points,y)
    return reg.coef_
def distance_point_line(point,line):#gives the distance between a plane and a point
    return abs(line[0]*point[0]-point[1]+line[1])/sqrt(pow(line[0],2)+1)

def mean_distance_batch_line(batch,line):#gives the distance between a plane and a point
    mean=0
    for point in batch:
        mean+=distance_point_line(point,line)
    return mean/len(batch)

def linkBatch_by_line(index,data_by_wire,linkage_threshold=0.5):#link batch based on the proximity of the points to the other's bach plane
    line=findLine(data_by_wire[index])
    #print(line)
    for i in range(len(data_by_wire)):
        if mean_distance_batch_line(data_by_wire[i],line)<linkage_threshold and i!=index:
            for point in data_by_wire[i]:
                data_by_wire[index].append(point)
            data_by_wire.pop(i)
            return data_by_wire,True
    return [],False

def agglomerateWithline(data_by_wire):#make plane based agglomeration
    sizes=sort(data_by_wire)
    for size in sizes:
        if size > 25:
            for i in range(len(data_by_wire)):
                if len(data_by_wire[i])==size:
                    res=linkBatch_by_line(i,data_by_wire)
                    if res[1] == True:
                        return agglomerateWithline(res[0])
    return data_by_wire