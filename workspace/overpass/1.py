import overpy

api = overpy.Overpass()

'''
way(50.0753397, 14.4189888, 50.0795436, 14.3907308)["highway"]["motorway"];
(._; >;);

bbox= 14.418654,50.077139,14.422517,50.078433  50.077167    
'''

# fetch all ways and nodes
result = api.query("node(50.077300,14.424684,50.077529,14.421268);out;")

for node in result.nodes:
    print("    Lat: %f, Lon: %f" % (node.lat, node.lon))
