from sklearn.decomposition import PCA
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