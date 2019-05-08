import googlemaps
from datetime import datetime
import polyline
import google_streetview.api
import google_streetview.helpers

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

print(type(directions_result[0]['legs'][0]['steps'][0]))
print(directions_result[0]['legs'][0]['steps'])
for key,val in directions_result[0]['legs'][0]['steps'][0].items():
    print(key,'      ',val )

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

abc = polyline.decode(directions_result[0]['overview_polyline']['points'])
for i in range(0, len(abc)):
    location1 = str(abc[i])
    location += location1.strip(')(') + '; '

apiargs = {
    'location': '50.0753397,14.4189888 ; '
                '50.0795436,14.3907308 ;'
                '50.10291748018805, 14.39132777985096',
    'size': '640x640',
    'heading': '0;45;90;135;180;225;270',
    'fov': '90',
    'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
    'pitch': '-90;0;90'
}

# Get a list of all possible queries from multiple parameters
api_list = google_streetview.helpers.api_list(apiargs)

# Create a results object for all possible queries
results = google_streetview.api.results(api_list)

# # Preview results
# results.preview()
#
# Download images to directory 'downloads'
results.download_links('downloads')
#
# # Save metadata
# results.save_metadata('metadata.json')
#print(results.metadata[2]['location'])#['lat'])
print(len(results.metadata))