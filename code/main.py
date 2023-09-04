import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
    print(df)
if __name__ == '__main__' :
    main()
