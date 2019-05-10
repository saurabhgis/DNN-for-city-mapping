import googlemaps
from datetime import datetime
import polyline
import google_streetview.api
import google_streetview.helpers
from math import sqrt
import numpy as np
import  folium

maxdis = 0.0002
def euclid(s1,s2):
	square = pow((s1[0] - s2[0]),2) + pow((s1[1]-s2[1]),2)
	return sqrt(square)


gmaps = googlemaps.Client(key='AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo')

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Karlovo nám. 287/18, 120 00 Nové Město",
                                     "Karlovo nám. 319/3, 120 00 Nové Město",
                                     mode="driving",
                                     departure_time=now)
location1 = []
location = ''
abc = polyline.decode(directions_result[0]['overview_polyline']['points'])

for i in range(0, len(abc)-1):
    dis = euclid(abc[i],abc[i+1])
    if dis > maxdis:
        sam = dis / maxdis
        lat = np.linspace(abc[i][0], abc[i+1][0], sam)
        lng = np.linspace(abc[i][1],abc[i+1][1],sam)
        for k in range(1,len(lat)-1):
            location += str(lat[k]) + ',' + str((lng[k])) + ';'
    location1 = str(abc[i])
    location += location1.strip(')(') + '; '
print(len(location))

apiargs = {
    'location': location,
    'size': '640x640',
    'heading': '0',
    'fov': '120',
    'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
    'pitch': '0'
}

# Get a list of all possible queries from multiple parameters
api_list = google_streetview.helpers.api_list(apiargs)

# Create a results object for all possible queries
results = google_streetview.api.results(api_list)

results.download_links('downloads')

print(len(results.metadata))