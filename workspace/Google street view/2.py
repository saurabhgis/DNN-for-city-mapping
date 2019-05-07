import googlemaps
from datetime import datetime

'''
way(50.0753397, 14.4189888, 50.0795436, 14.3907308)
@50.0758579,14.3982505
@50.0759183,14.3983564
@50.0759785,14.3984617
'''
now = datetime.now()
gmaps = googlemaps.Client(key='AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo')

directions_result = gmaps.directions("50.0753397, 14.4189888",
                                     "50.0795436, 14.3907308",
                                     mode="transit",
                                     departure_time=now)

print(directions_result)