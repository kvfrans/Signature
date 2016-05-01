import urllib2

image = urllib2.urlopen('https://signatureauthentication.firebaseio.com/image.json').read()
image = image[1:-1]
image = image.replace("\\r\\n", "")
fh = open("imageToSaveCurl.txt", "wb")
fh.write(image)
fh.close()
# print image
fh = open("imageToSave.png", "wb")
fh.write(image.decode('base64'))
fh.close()
