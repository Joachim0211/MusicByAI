import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from myconfig import keys

auth_manager = SpotifyClientCredentials(client_id=keys['client_id'],
                                        client_secret=keys['client_secret'])
sp = spotipy.Spotify(auth_manager=auth_manager)

path = '.\\'#.\Data\\'

songs = pd.read_csv(path+'songs5000_clusters.csv')
#songs.set_index(["name", "artist"],inplace=True)
songs = songs.loc[:,['cluster','id','sh_score']]

cluster_genres=[]
for i in range(6):
    songs_cluster = songs[songs['cluster']==i]
    cluster_core = songs_cluster.sort_values(by=['sh_score'],ascending=False).head(20)
    genres = []
    for j in range(len(cluster_core['id'])):
        song_uri = 'spotify:track:'+cluster_core['id'].iloc[j].strip()
        sp_track = sp.track(song_uri)
        sp_artist = sp.artist(sp_track['artists'][0]['uri'])
        genres = genres + sp_artist['genres']
    #print(genres)
    g_counts={}
    for s in genres:
        g_counts[s]=genres.count(s)
    g_counts_df = pd.DataFrame([{"name": key, "count": value} for key, value in g_counts.items()])
    g_counts_df = g_counts_df.sort_values('count',ascending=False).head(5)
    g_counts_df['cluster']=i
    cluster_genres = cluster_genres + [g_counts_df]

cluster_genres_df=pd.concat(cluster_genres).reset_index()
cluster_genres_df.to_csv('claster_genres6.csv')
#birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
#results = sp.artist_albums(birdy_uri, album_type='album')
##song_uri = 'spotify:track:1n7JnwviZ7zf0LR1tcGFq7'
#results = sp.audio_analysis(song_uri)
##results = sp.track(song_uri)
##genres = sp.artist(results['artists'][0]['uri'])['genres']

