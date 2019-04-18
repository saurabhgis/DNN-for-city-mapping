location1 =[]
location = ''
for i in range(0,20):
    lat = str(50.0753397+i)
    lon = str(14.4189888+i)
    location1 = lat+','+ lon +';'
    location += location1

apiargs = {'location':location,
  # 'location': '50.0753397,14.4189888',
  'size': '640x640',
  'heading': '0;45;90;135;180;225;270',
  'fov': '90',
  'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
  'pitch': '-90;0;90'
}

apiargs1 = {
  'location': '50.0753397,14.4189888 ; 50.0795436,14.3907308 ;50.10291748018805, 14.39132777985096',
  'size': '640x640',
  'heading': '0;45;90;135;180;225;270',
  'fov': '90',
  'key': 'AIzaSyCciJlgQzOFXZXhvM1ORscu0Cj5dj-lTRo',
  'pitch': '-90;0;90'
}
#print(location)
print(apiargs)
print(apiargs1)