import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as skl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


path = '..\Data\\'

#songs=pd.read_csv(path+'df_audio_features_10', index_col=["song_name", "artist"])
#songs=pd.read_csv(path+'df_audio_features_1000', index_col=["name", "artist"])
songs=pd.read_csv(path+'df_audio_features_5000')
songs.columns=songs.columns.str.strip()
songs.set_index(["name", "artist"],inplace=True)

songs = songs.loc[(songs['loudness']>-45)&(songs['speechiness']<0.6),:]
songs_slice = songs.loc[:,['danceability', 'energy', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'time_signature']]

#scaler = skl.RobustScaler()
#scaler = skl.QuantileTransformer()
scaler = skl.MinMaxScaler(feature_range=(0,1))
scaler.fit(songs_slice)
songs_scaled = scaler.transform(songs_slice)

cluster_n = []
inert = []
sh_score = []
for n_cl in range(2,22):
    my_kmeans = KMeans(n_clusters= n_cl, random_state = 1)
    my_kmeans.fit(songs_scaled)
    cluster_n = cluster_n + [n_cl]
    inert = inert + [my_kmeans.inertia_]
    sh_score = sh_score + [silhouette_score(songs_scaled, my_kmeans.labels_)]

plt.figure(1)    
plt.plot(cluster_n,inert,'bo-')
plt.figure(2)
plt.plot(cluster_n,sh_score,'ro-')

