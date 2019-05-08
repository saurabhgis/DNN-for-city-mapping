import overpy

api = overpy.Overpass()

# fetch all ways and nodes
result = api.query("""[out:json][timeout:25];
// gather results
(
  // query part for: “highway=*”

  way["highway"](50.077,14.38,50.083,14.391);

);
// print results
out body;
>;
out skel qt;""")
print(len(result.nodes))
for node in result.nodes:
    print("    Lat: %f, Lon: %f" % (node.lat, node.lon))
