import cv2
 
# read image
img = cv2.imread('./Images/angelamerkel_4.jpg', cv2.IMREAD_UNCHANGED)
 
# get dimensions of image
dimensions = img.shape

print('Image Dimension    : ',dimensions)
resized = img
if int(img.shape[1]) > 450 :
    scale_percent = (450*100)/img.shape[1]
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)

# get dimensions of image
dimensions = resized.shape

print('Image Dimension    : ',dimensions)

cv2.imshow("Resized image", resized)
cv2.waitKey(0)
waitKey(0)
cv2.destroyAllWindows()
