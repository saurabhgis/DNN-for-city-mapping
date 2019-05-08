# Import google_streetview for the api and helper module
import google_streetview.api
import google_streetview.helpers
import overpy
api = overpy.Overpass()
#  'location': '46.414382,10.013988;40.720032,-73.988354',
# Create a dictionary with multiple parameters separated by ;

# resultov = api.query("""way
# ["name"="Nad Závěrkou"]
# (50.07,14.0,50.8,14.4);
# /*added by auto repair*/
# (._;>;);
# /*end of auto repair*/
# out;""")
resultov = api.query("""[out:json][timeout:25];
// gather results
(
  // query part for: “highway=*”
  way["highway"](50.745,7.17,50.75,7.18);
);
// print results
out body;
>;
out skel qt;""")

location1 = []
location = ''
print(resultov.ways)
for node in resultov.nodes:
    print("    Lat: %f, Lon: %f" % (node.lat, node.lon))

# for ways in resultov.ways:
#     print("    Lat: %f, Lon: %f" % (node.lat, node.lon))
    # lat = str(node.lat)
    # lon = str(node.lon)
    # location1 = lat + ',' + lon + '; '
    # location += location1

# apiargs = {
#   'location': location,
#   'size': '640x640',
#   'heading': '270',
#   'fov': '90',
#   'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
#   'pitch': '0'
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