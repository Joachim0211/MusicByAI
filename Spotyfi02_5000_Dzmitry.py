import pandas as pd
import numpy as np
#from sklearn.preprocessing import MinMaxScaler
import sklearn.preprocessing as skl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


path = '..\Data\\'

#songs=pd.read_csv(path+'df_audio_features_10', index_col=["song_name", "artist"])
#songs=pd.read_csv(path+'df_audio_features_1000', index_col=["name", "artist"])
songs=pd.read_csv(path+'df_audio_features_5000')
songs.columns=songs.columns.str.strip()
songs.set_index(["name", "artist"],inplace=True)


songs_slice = songs.loc[:,['danceability', 'energy', 'loudness', 'mode', 'speechiness',
       'acousticness', 'instrumentalness', 'liveness', 'valence', 'time_signature']]

#scaler = skl.RobustScaler()
#scaler = skl.QuantileTransformer()
scaler = skl.MinMaxScaler(feature_range=(0,1))
scaler.fit(songs_slice)
songs_scaled = scaler.transform(songs_slice)
 
songs_scaled_df = pd.DataFrame(songs_scaled,
             index=songs_slice.index,
             columns=songs_slice.columns)

my_kmeans = KMeans(n_clusters= 5)
my_kmeans.fit(songs_scaled)
clusters = my_kmeans.labels_
inert = my_kmeans.inertia_
#clusters1 = my_kmeans.predict(songs_scaled)
songs["cluster"] = clusters
songs_scaled_df["cluster"] = clusters
songs_scaled_df["html"] = songs["html"]

songs_scaled_df.sort_values(by=['cluster'], inplace=True)

clusters_df = songs_scaled_df.groupby('cluster').agg('mean')

# plt.plot(songs_scaled_df.columns,np.array(songs_scaled_df).transpose(),'bo',alpha=0.01)

songs_scaled_df.plot(subplots=True,marker='.',linestyle='none')
#songs_scaled_df.loc[:,['danceability', 'energy', 'loudness', 'mode', 'speechiness','cluster']].plot(subplots=True,marker='.',linestyle='none')
#songs_scaled_df.loc[:,['acousticness', 'instrumentalness', 'liveness', 'valence', 'time_signature','cluster']].plot(subplots=True,marker='.',linestyle='none')
ax = plt.gca()
ax.axes.xaxis.set_ticklabels([])
plt.tight_layout()



# songs_scaled_df.hist()


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














