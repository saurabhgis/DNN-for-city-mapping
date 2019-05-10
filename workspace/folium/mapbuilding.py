import folium


m = folium.Map(
    location=[45.372, -121.6972],
    zoom_start=12,
    tiles='Stamen Terrain')
location = [(44,-121),(45,-121),(45,-122),(45,-120) ]

for i in location:
    folium.Marker(
        location=i,
        popup='Mt. Hood Meadows',
        icon=folium.Icon(icon='home')
    ).add_to(m)

m.save(outfile='map.html')