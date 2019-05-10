import googlemaps
from datetime import datetime
import polyline
import google_streetview.api
import google_streetview.helpers
from math import sqrt
import numpy as np
import  folium

maxdis = 0.0005
def euclid(s1,s2):
	square = pow((s1[0] - s2[0]),2) + pow((s1[1]-s2[1]),2)
	return sqrt(square)



def sampless(s1,s2,dist):
    sam = dist/maxdis
    print('sample' ,sam)
    lat = np.linspace(s1[0], s1[1], sam)
    lng = np.linspace(s2[0],s2[1], sam)
    return  lat ,lng


    # for lon in range(s2[0],s2[1] , )
    # print('s')
#  'location': '46.414382,10.013988;40.720032,-73.988354',
# Create a dictionary with multiple parameters separated by ;

gmaps = googlemaps.Client(key='AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo')

# Request directions via public transit
now = datetime.now()
directions_result = gmaps.directions("Dejvice, Prague 6",
                                     "Charles Square, Nové Město, Praha",
                                     mode="driving",
                                     departure_time=now)
#print(directions_result[0]['overview_polyline']['points'])
#
# print(type(directions_result[0]['legs'][0]['steps'][0]))
# print(directions_result[0]['legs'][0]['steps'])
# for key,val in directions_result[0]['legs'][0]['steps'][0].items():
#     print(key,'      ',val )

#print(directions_result[0]['bounds'])
# abc = polyline.decode(directions_result[0]['overview_polyline']['points'])

# print(abc[0])
# print(len(abc))
# location1 = []
# location = ''
# for i in range(0,len(abc)):
#     location1 = str(abc[i])
#     location += location1.strip(')(') + '; '
# print(location)
# for key in directions_result[0]:
#     print(key,'\n')

location1 = []
location = ''
map = folium.Map(location=[50.0753397, 14.4189888])
abc = polyline.decode(directions_result[0]['overview_polyline']['points'])
#print(np.linspace(abc[0][0], abc[1][0], 10))
j=0
for i in range(0, len(abc)-1):
    dis = euclid(abc[i],abc[i+1])
    #print(dis ,'disatnce ')
    if dis > 0.0005:
        j=j+1
        sam = dis / maxdis
        #print("hahahah")
        lat = np.linspace(abc[i][0], abc[i+1][0], sam)
        lng = np.linspace(abc[i][1],abc[i+1][1],sam)
        # print(lat,lng)
        for k in range(0,len(lat)-1):
            location += str(lat[k]) + ',' + str((lng[k])) + ';'
            loc = [lat[k],lng[k]]
            folium.Marker(
                location=loc,
                popup='waypoint',
                icon=folium.Icon(color='white')
            ).add_to(map)
    location1 = str(abc[i])
    folium.Marker(
        location=abc[i],
        popup='waypoint',
        icon=folium.Icon(color='white')
    ).add_to(map)

    location += location1.strip(')(') + '; '
print(location)

map.save(outfile='map.html')

# apiargs = {
#     'location': '50.0753397,14.4189888 ; '
#                 '50.0795436,14.3907308 ;'
#                 '50.10291748018805, 14.39132777985096',
#     'size': '640x640',
#     'heading': '0;45;90;135;180;225;270',
#     'fov': '90',
#     'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
#     'pitch': '-90;0;90'
# }
#
# # Get a list of all possible queries from multiple parameters
# api_list = google_streetview.helpers.api_list(apiargs)
#
# # Create a results object for all possible queries
# results = google_streetview.api.results(api_list)
#
# # # Preview results
# # results.preview()
# #
# # Download images to directory 'downloads'
# results.download_links('downloads')
# #
# # # Save metadata
# # results.save_metadata('metadata.json')
# #print(results.metadata[2]['location'])#['lat'])
# print(len(results.metadata))