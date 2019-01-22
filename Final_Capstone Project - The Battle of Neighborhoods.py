
# coding: utf-8

# # Capstone Project - The Battle of Neighborhoods
# 

# ### Project Description

# A coffee house may be one of the more simple businesses to start among businesses in the food and beverage industry. The ability to plan ahead and meet the needs of customers will help create a successful business that will endure time. The location should be enegry efficient. This project recommends location to an entrepreneur to open a coffee shop in New York City using data science. 

# Whenever people want to open a new shop, they explore the place and try to fetch as much information as possible around it. It can be the neighborhood, venues, etc., This is can be termed as request for a search algorithm which usually returns the requested features such as population rate, schools/colleges/offices around, weather conditions, recreational facilities etc.  It would be beneficial to have an application which could make easy by considering a comparative analysis between the neighborhood with provided factors.

# ### Data Section

# New York City Neighborhood Names point file from https://geo.nyu.edu/catalog/nyu_2451_34572. It has a total of 5 boroughs and 306 neighborhoods. 
# 
# #### Foursquare API:
# It has a database of more than 105 million places. This project would use Four-square API as its prime data gathering source.

# #### Python Library Files :
#     • Pandas - Library for Data Analysis 
#     • NumPy – Library to handle data in a vectorized manner
#     • JSON – Library to handle JSON files 
#     • Folium – Map rendering Library
#     • Matplotlib – Python Plotting Module 
#     • Geopy – To retrieve Location Data 
#     • Requests – Library to handle http requests
#     • Sklearn – Python machine learning Library 

# #### Folium:
# Python visualization library would be used to visualize the neighborhoods cluster distribution of Chicago city over an interactive leaflet map.Extensive comparative analysis of two randomly picked neighborhoods world be carried out to derive the desirable insights from the outcomes using python’s scientific libraries Pandas, NumPy and Scikit-learn.
# 
# #### Unsupervised machine learning algorithm:
# K-mean clustering would be applied to form the clusters of different categories of places  in and around the neighborhoods. Each of them would be analyzed individually and comparatively to derive the best location.

# ## Implemetation

# ### Download and Expolre Data Set

# In[2]:



import numpy as np # library to handle data in a vectorized manner

import pandas as pd # library for data analsysis
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

import json # library to handle JSON files

get_ipython().system(u"conda install -c conda-forge geopy --yes # uncomment this line if you haven't completed the Foursquare API lab")
from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

get_ipython().system(u"conda install -c conda-forge folium=0.5.0 --yes # uncomment this line if you haven't completed the Foursquare API lab")
import folium # map rendering library

print('Libraries imported Successfully.')


# #### Neighborhood has a total of 5 boroughs and 306 neighborhoods. In order to segement the neighborhoods and explore them, we will  need a dataset that contains the 5 boroughs and the neighborhoods with latitude and logitude coordinates of each neighborhood. 

# In[3]:


get_ipython().system(u"wget -q -O 'newyork_data.json' https://ibm.box.com/shared/static/fbpwbovar7lf8p5sgddm06cgipa2rxpe.json")
print('Data downloaded!')


# In[4]:


with open('newyork_data.json') as json_data:
    newyork_data = json.load(json_data)


# In[5]:


newyork_data


# In[6]:


neighborhoods_data = newyork_data['features']


# In[7]:


neighborhoods_data[0]


# In[8]:


# define the dataframe columns
column_names = ['Borough', 'Neighborhood', 'Latitude', 'Longitude'] 

# instantiate the dataframe
            neighborhoods = pd.DataFrame(columns=column_names)


# In[9]:


for data in neighborhoods_data:
    borough = neighborhood_name = data['properties']['borough'] 
    neighborhood_name = data['properties']['name']
        
    neighborhood_latlon = data['geometry']['coordinates']
    neighborhood_lat = neighborhood_latlon[1]
    neighborhood_lon = neighborhood_latlon[0]
    
    neighborhoods = neighborhoods.append({'Borough': borough,
                                          'Neighborhood': neighborhood_name,
                                          'Latitude': neighborhood_lat,
                                          'Longitude': neighborhood_lon}, ignore_index=True)


# In[10]:


neighborhoods.head()


# In[13]:


print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)


# In[14]:


address = 'New York City, NY'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of New York City are {}, {}.'.format(latitude, longitude))


# In[16]:


# create map of New York using latitude and longitude values
map_newyork = folium.Map(location=[latitude, longitude], zoom_start=10)

# add markers to map
for lat, lng, borough, neighborhood in zip(neighborhoods['Latitude'], neighborhoods['Longitude'], neighborhoods['Borough'], neighborhoods['Neighborhood']):
    label = '{}, {}'.format(neighborhood, borough)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=1,
        parse_html=False).add_to(map_newyork)  
    
map_newyork


# In[17]:


manhattan_data = neighborhoods[neighborhoods['Borough'] == 'Manhattan'].reset_index(drop=True)
manhattan_data.head()


# In[19]:


address = 'Manhattan, NY'

geolocator = Nominatim()
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Manhattan are {}, {}.'.format(latitude, longitude))


# In[20]:


# create map of Manhattan using latitude and longitude values
map_manhattan = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(manhattan_data['Latitude'], manhattan_data['Longitude'], manhattan_data['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        parse_html=False).add_to(map_manhattan)  
    
map_manhattan


# In[21]:


CLIENT_ID = 'QACUHLIWC30ZYBY5FL1MLPNWPMBY21KBKTNRLAMIK055MV0Y' # your Foursquare ID
CLIENT_SECRET = 'BWO3PLMERVP2MU0STUPDC2JRU2SBOAY3SUJ1TNINIBJFLF3V' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version


# In[22]:


manhattan_data.loc[0, 'Neighborhood']


# In[23]:


neighborhood_latitude = manhattan_data.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = manhattan_data.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = manhattan_data.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))


# In[24]:


# type your answer here


#The correct answer is:
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius
# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL


# In[25]:


results = requests.get(url).json()
results


# In[26]:


# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']


# venues = results['response']['groups'][0]['items']
#     
# nearby_venues = json_normalize(venues) # flatten JSON
# 
# # filter columns
# filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
# nearby_venues =nearby_venues.loc[:, filtered_columns]
# 
# # filter the category for each row
# nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)
# 
# # clean columns
# nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]
# 
# nearby_venues.head()

# In[28]:


print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))


# #### Explore neighborhood in Manhattan

# In[30]:


def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)


# In[31]:


manhattan_venues = getNearbyVenues(names=manhattan_data['Neighborhood'],
                                   latitudes=manhattan_data['Latitude'],
                                   longitudes=manhattan_data['Longitude']
                                  )


# In[32]:


print(manhattan_venues.shape)
manhattan_venues.head()


# In[33]:


manhattan_venues.groupby('Neighborhood').count()


# In[34]:


print('There are {} uniques categories.'.format(len(manhattan_venues['Venue Category'].unique())))


# #### Analyse each neighborhood

# In[42]:


# one hot encoding
manhattan_onehot = pd.get_dummies(manhattan_venues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
manhattan_onehot['Neighborhood'] = manhattan_venues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [manhattan_onehot.columns[-1]] + list(manhattan_onehot.columns[:-1])
manhattan_onehot = manhattan_onehot[fixed_columns]

manhattan_onehot.head()


# In[44]:


manhattan_onehot.shape


# In[45]:


manhattan_grouped = manhattan_onehot.groupby('Neighborhood').mean().reset_index()
manhattan_grouped


# In[46]:


num_top_venues = 5

for hood in manhattan_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = manhattan_grouped[manhattan_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# In[47]:


def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]


# In[48]:


num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = manhattan_grouped['Neighborhood']

for ind in np.arange(manhattan_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(manhattan_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted


# #### Cluster Neighbor

# In[49]:


# set number of clusters
kclusters = 5

manhattan_grouped_clustering = manhattan_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(manhattan_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 


# In[92]:


manhattan_merged = manhattan_data

# add clustering labels
manhattan_merged['Cluster Labels'] = kmeans.labels_

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
manhattan_merged = manhattan_merged.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

manhattan_merged # check the last columns!


# In[51]:


# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i+x+(i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(manhattan_merged['Latitude'], manhattan_merged['Longitude'], manhattan_merged['Neighborhood'], manhattan_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters


# #### Cluster1

# In[60]:


cluster1 = manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 0, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
cluster1


# #### Cluster 2

# In[61]:


cluster2 = manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 1, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
cluster2


# #### Cluster 3

# In[62]:


cluster3 = manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 2, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
cluster3


# #### Cluster 4

# In[64]:


cluster4 = manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 3, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
cluster4


# #### Cluster 5

# In[65]:


cluster5 = manhattan_merged.loc[manhattan_merged['Cluster Labels'] == 4, manhattan_merged.columns[[1] + list(range(5, manhattan_merged.shape[1]))]]
cluster5


# In[66]:


clusters=pd.DataFrame({"Cluster 1":cluster1["Neighborhood"],
                      "Cluster 2":cluster2["Neighborhood"],
                      "Cluster 3":cluster3["Neighborhood"],
                      "Cluster 4":cluster4["Neighborhood"],
                      "Cluster 5":cluster5["Neighborhood"]
                       })


# In[67]:


clusters = clusters.replace(np.nan, '', regex=True)


# In[68]:


clusters


# In[95]:


print ("Neighborhood has more vistors to Coffee shop are\n")
manhattan_merged[manhattan_merged['1st Most Common Venue'] == 'Coffee Shop']


# ## Conclusion :

# This analysis concludes that 9 neighborhoods having most frequent vistors to coffeeshop and are preferable location to open a new coffee shop
