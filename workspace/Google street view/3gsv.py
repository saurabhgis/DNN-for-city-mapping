'''

'50.0753397,14.4189888 ; '
              '50.0795436,14.3907308 ;'
'''

#  'location': '46.414382,10.013988;40.720032,-73.988354',

# Import google_streetview for the api and helper module
import google_streetview.api
import google_streetview.helpers
import math

def get_x_y_co(circles):
    xc = circles[0] #x-co of circle (center)
    yc = circles[1] #y-co of circle (center)
    r = circles[2] #radius of circle
    arr=[]
    for i in range(0,180,30):
        y = yc + r*math.cos(i)
        x = xc+  r*math.cos(i)
        x=float(x)
        y=float(y)
        #Create array with all the x-co and y-co of the circle

        arr.append([x,y])
    return arr

def runquery(location):
    apiargs = {
      'location': location,
      'size': '640x640',
      'heading': '0',
      'fov': '90',
      'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
      'pitch': '0'
    }
    # Get a list of all possible queries from multiple parameters
    api_list = google_streetview.helpers.api_list(apiargs)
    # Create a results object for all possible queries
    results = google_streetview.api.results(api_list)
    print('length of result',len(results.metadata))
    return results

# # Create a dictionary with multiple parameters separated by ;
# location1 = []
# location = ''
# lat = str(50.103046)
# lon = str(14.426656)
# location = lat + ',' + lon + '; '

def check_result(results):
    location = ''
    location1 = []
    for i in range(len(results.metadata)):
        if results.metadata[i]['status'] == 'OK':
            print('image available at ', results.metadata[i]['location'])
            lat = float(results.metadata[0]['location']['lat']) + 0.00009
            lon = float(results.metadata[0]['location']['lon']) + 0.00009
            location = str(lat) + ',' + str(lon) + '; '
            # Download images to directory 'downloads'
            results.download_links('downloads')
        if results.metadata[i]['status'] is not 'OK':
            print('searching for path')
            abc = get_x_y_co([50.112046, 14.435656, 0.005])
            for j in range(0,6):
                lat = str(abc[j][0])
                lon = str(abc[j][1])
                location1 = lat + ',' + lon + '; '
                location += location1
            print(location)

def download_image(results):
    results.download_links('downloads')