# Import google_streetview for the api and helper module
import google_streetview.api
import google_streetview.helpers
import json



'''
'50.0753397,14.4189888 ; 50.0795436,14.3907308 ;50.10291748018805, 14.39132777985096', 
'''
location1 =[]
location = ''
for i in range(0,2):
    lat = str((50.0753397- i*0.0001))
    lon = str(14.4189888)
    location1 = lat+','+ lon +'; '
    location += location1

#  'location': '46.414382,10.013988;40.720032,-73.988354',
#'50.0795436,14.3907308 ;'
#'50.10291748018805, 14.39132777985096'
# Create a dictionary with multiple parameters separated by ;

apiargs = {
#  'location': location,
  'location':'50.081584, 14.390405 ;50.082674, 14.386711',
  'size': '640x640',
  'heading': '180',
  'fov': '90',
  'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
  'pitch': '0'
}

#print(data[1])
#print(apiargs)
# Get a list of all possible queries from multiple parameters
api_list = google_streetview.helpers.api_list(apiargs)

# Create a results object for all possible queries
results = google_streetview.api.results(api_list)

# Preview results
# results.preview()

# Download images to directory 'downloads'
# results.download_links('downloads')

#Save metadata
results.save_metadata('metadata.json')

jsonfile = 'D:\Projects\DNN for city mapping\workspace\polygonimage\metadata.json'

with open(jsonfile) as f:
    data = json.load(f)

print(data[1]['status'])