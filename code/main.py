import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

def plot(df):#a function to visualise data frame
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for value,point in df.iterrows():
        xs = point['x']
        ys = point['y']
        zs = point['z']
        ax.scatter(xs, ys, zs)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def main():  
    df = pd.read_parquet('../data/lidar_cable_points_easy.parquet') #extracting data into df
    #plot(df)
    #print(df)
    data_numpy_array = df.to_numpy()#create a numpy array from data frame
    clustering = AgglomerativeClustering(n_clusters=None,linkage='single',distance_threshold=0.75).fit(data_numpy_array)#link points/agglomerations when the minimum distance between the points/aglomerations is inferior to distance_threshold with all the points in data_numpy_array
    #print("there are "+str(clustering.n_clusters_)+" wires detected by clustering") 

if __name__ == '__main__' :
    main()
