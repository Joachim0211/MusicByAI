import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
#from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as skl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
import plotly.express as px
from spider_plot import radar_chart


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
#scaler = skl.StandardScaler()
scaler.fit(songs_slice)
songs_scaled = scaler.transform(songs_slice)
 
songs_scaled_df = pd.DataFrame(songs_scaled,
             index=songs_slice.index,
             columns=songs_slice.columns)

my_kmeans = KMeans(n_clusters= 40, random_state = 1)
my_kmeans.fit(songs_scaled)
clusters = my_kmeans.labels_
inert = my_kmeans.inertia_ 

# songs_scaled_df.hist()

songs["cluster"] = clusters
songs_scaled_df["cluster"] = clusters
songs_scaled_df["html"] = songs["html"]
songs_scaled_df["id"] = songs["id"]

songs_by_cluster = songs_scaled_df.sort_values(by=['cluster']).copy()

songs_by_cluster.plot(subplots=True,marker='.',linestyle='none')
#songs_scaled_df.loc[:,['danceability', 'energy', 'loudness', 'mode', 'speechiness','cluster']].plot(subplots=True,marker='.',linestyle='none')
#songs_scaled_df.loc[:,['acousticness', 'instrumentalness', 'liveness', 'valence', 'time_signature','cluster']].plot(subplots=True,marker='.',linestyle='none')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.tight_layout()
    

###########################################


# Euclidean (by default)
# ecl_dis = pd.DataFrame(pairwise_distances(songs_scaled_df), index=songs_scaled_df.index, columns=songs_scaled_df.index)
# plt.subplots(figsize=(12, 8))
# sns.heatmap(ecl_dis)

###########################################
songs_scaled_df["sh_score"] = silhouette_samples(songs_scaled, my_kmeans.labels_)
songs_scaled_df.to_csv('songs5000_40clusters.csv')

songs_scaled_df1 = songs_scaled_df.loc[songs_scaled_df["sh_score"] > 0.2,:]
#songs_scaled_df["sh_score"].hist()

################################
# cl_pos = pd.DataFrame(my_kmeans.cluster_centers_)
# cl_pos.columns = ['danceability', 'energy', 'loudness', 'mode', 'speechiness',
#         'acousticness', 'instrumentalness', 'liveness', 'valence', 'time_signature']
# radar_chart(cl_pos)


#plot each column vs one another
################################
# cols = ['danceability', 'energy', 'loudness', 'speechiness',
#        'acousticness', 'instrumentalness', 'liveness', 'valence']
# i=0
# for col1 in cols:
#     for col2 in cols:
#         print(col1+' vs '+col2)
#         if (i==0)|(i==16)|(i==32)|(i==48)|(i==64):
#             fig, axs = plt.subplots(4, 4, figsize=(10, 6), constrained_layout=True)
#             j=0
#         ax=axs.flat[j]
#         ax.set_title(col2+' vs '+col1)
#         #ax.plot(songs_scaled_df[col1], songs_scaled_df[col2], '.', ls='', ms=4)
#         sns.scatterplot(data=songs_scaled_df, x=col1, 
#                         y=col2, hue = "cluster", ax=ax, 
#                         legend=False,marker=".", palette = "Set1")
#         i=i+1
#         j=j+1














