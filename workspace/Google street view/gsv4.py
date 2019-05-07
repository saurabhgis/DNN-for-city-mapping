import google_streetview.api
import google_streetview.helpers
import math
availocation = ''
location = ''
def get_x_y_co(circles):
    xc = circles[0] #x-co of circle (center)
    yc = circles[1] #y-co of circle (center)
    r = circles[2] #radius of circle
    locationtest =''
    for i in range(0,360,30):
        y = yc + r*math.cos(i)
        x = xc+  r*math.cos(i)
        lat = str(x)
        lon = str(y)
        location1 = lat + ',' + lon + '; '
        locationtest += location1
    return locationtest

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
    # print('length of result',len(results.metadata))
    return results

def check_result(results):
    availocation = ''
    newlocation = []
    for i in range(len(results.metadata)):
        if results.metadata[i]['status'] == 'OK':
            newlocation = [lat ,lon]
            availocation = availocation + str(lat) + ',' + str(lon) + '; '
    return availocation , newlocation

def download_image(location):
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
    results.download_links('downloads')

if __name__ == '__main__':
    location1 = []
    locations = ''
    lat = str(50.103046)
    lon = str(14.426656)
    location = str(50.103046) + ',' + str(14.426656) + '; '
    for i in range(0, 4):
        results = runquery(location)
        availocation , locations = check_result(results)
        location = get_x_y_co([float(locations[0]), float(locations[1]), 0.09])

