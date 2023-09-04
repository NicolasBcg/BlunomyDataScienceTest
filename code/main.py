import pandas as pd

def main():
    df = pd.read_parquet('../data/lidar_cable_points_easy.parquet')
    print(df.head())
if __name__ == '__main__' :
    main()