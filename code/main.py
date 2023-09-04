import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering

colors=['#808080', '#FF0000', '#FFFF00', '#00FF00', '#0000FF', '#FF00FF', '#C0C0C0', '#FFA500', '#191970', '#17becf', '#FF69B4', '#8B008B', '#6B8E23', '#00BFFF'] #a bunch of color to display different labels

def plot(array,number_of_labels=0):#a function to visualise 3D area. no number_of_label if there are no labels
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    if number_of_labels==0:
        for point in array:
            ax.scatter(point[0], point[1], point[2])
    else: 
        if number_of_labels > len(colors):
            for _ in range(len(colors),number_of_labels+1):
                colors.append('#000000')
        for label_number in range(number_of_labels):
            for point in array[label_number]:
                ax.scatter(point[0], point[1], point[2],c=colors[label_number])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

def cluster(data_numpy_array):
    clustering = AgglomerativeClustering(n_clusters=None,linkage='single',distance_threshold=0.75).fit(data_numpy_array)#link points/agglomerations when the minimum distance between the points/aglomerations is inferior to distance_threshold with all the points in data_numpy_array
    labels=[[label] for label in clustering.labels_]#putting label on the right format to concatenate

    #reorganizing datas into  [[points from wire 1],...,[points from wire n]]
    data_by_label=[[] for _ in range(clustering.n_clusters_)]
    for i in range(len(clustering.labels_)):
        data_by_label[clustering.labels_[i]].append(data_numpy_array[i])
    return data_by_label


def main():  
    df = pd.read_parquet('../data/lidar_cable_points_easy.parquet') #extracting data into df
    data_numpy_array = df.to_numpy()#create a numpy array from data frame
    #plot(data_numpy_array)#plot array
    data_by_label = cluster(data_numpy_array)
    print("there are "+str(len(data_by_label))+" wires detected by clustering") 
    plot(data_by_label,len(data_by_label))

if __name__ == '__main__' :
    main()
