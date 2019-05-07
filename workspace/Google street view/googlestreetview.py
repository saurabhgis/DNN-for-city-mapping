# Import google_streetview for the api and helper module
import google_streetview.api
import google_streetview.helpers

#  'location': '46.414382,10.013988;40.720032,-73.988354',
# Create a dictionary with multiple parameters separated by ;
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