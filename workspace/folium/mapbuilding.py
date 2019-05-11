import folium

col = ['green','red','blue','black']
# print(col[4])
m = folium.Map(
    location=[45.372, -121.6972],
    zoom_start=12,
    tiles='Stamen Terrain')
location = [(44,-121),(45,-121),(45,-122),(45,-120) ]
c =0
for i in location:
    folium.Marker(
        location=i,
        popup='something',
        icon=folium.Icon(color=col[c])
    ).add_to(m)
    c= c+1

m.save(outfile='map.html')
