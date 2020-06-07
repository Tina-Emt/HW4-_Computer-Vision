# -*- coding: utf-8 -*-

import numpy as np
import cv2
from matplotlib import pyplot as plt

#%% Part 1 A
# ORB Method
import numpy as np
import cv2


img1 = cv2.imread('image.jpg')    # queryImage
img2 = cv2.imread('template.jpg') # trainImage


# Initiate detector
orb = cv2.ORB_create()




kp1, des1 = orb.detectAndCompute(img1,None)



kp2, des2 = orb.detectAndCompute(img2,None)



# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 20 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

cv2.imshow('ORB METHOD', img3)

#%%
import numpy as np
import cv2


img1 = cv2.imread('image.jpg')    # queryImage
img2 = cv2.imread('template.jpg') # trainImage


# Initiate SIFT detector
akz = cv2.AKAZE_create()




kp1, des1 = akz.detectAndCompute(img1,None)



kp2, des2 = akz.detectAndCompute(img2,None)



# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)

cv2.imshow('AKAZE METHOD', img3)

#%% Part 1 B
# SIFT Method
import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('image.jpg',0)          # queryImage
img2 = cv2.imread('template.jpg',0) # trainImage

# Initiate SIFT detector
sift = cv2.SIFT()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None, flags=2)

plt.imshow(img3),plt.show()



#%%


def mouseListenerForFirstImage(event, x, y, flags, params):
    
    global points, clickedPoints1
    
    if event == cv2.EVENT_LBUTTONDOWN:
        srcPoints1.append((x, y))
        clickedPoints1 += 1
        cv2.circle(image1,(x, y), 1, (0,255,0), cv2.FILLED)

def mouseListenerForsecondImage(event, x, y, flags, params):
    
    global points, clickedPoints2
    
    if event == cv2.EVENT_LBUTTONDOWN:
        srcPoints2.append((x, y))
        clickedPoints2 += 1
        cv2.circle(image2,(x, y), 2, (255,0,0), cv2.FILLED)
        
#%% Part 2


import cv2
import numpy as np

image1 = cv2.imread("1-1.jpg")

row,column = image1.shape[:2]
row,column = int(row*0.5), int(column*0.5)

image1 = cv2.resize(image1, (column, row), interpolation = cv2.INTER_AREA)

clickedPoints1 = 0
srcPoints1 = []
counter = 4


while clickedPoints1 != counter:
    cv2.imshow("Image 1-1", image1)
    cv2.setMouseCallback("Image 1-1", mouseListenerForFirstImage)
    cv2.waitKey(1)
  
cv2.destroyAllWindows()

srcPoints1 = np.array(srcPoints1)
dstPoints1 = np.array([[550,400], [600,400], [600,450], [550,450]])

# finding homography
homographyMatrix,_ = cv2.findHomography(srcPoints1, dstPoints1)
# warp image 1 
output = cv2.warpPerspective(image1, homographyMatrix, (1000,750))

cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()



    


#%% Part 2 Second Image 
import cv2
import numpy as np

image2 = cv2.imread("1-2.jpg")

row,column = image2.shape[:2]
row,column = int(row*0.5), int(column*0.5)

image2 = cv2.resize(image2, (column, row), interpolation = cv2.INTER_AREA)

clickedPoints2 = 0
srcPoints2 = []


while clickedPoints2 != counter:
    cv2.imshow("Image 2-1", image2)
    cv2.setMouseCallback("Image 2-1", mouseListenerForsecondImage)
    cv2.waitKey(1)
  
cv2.destroyAllWindows()

srcPoints2 = np.array(srcPoints1)
dstPoints2 = np.array([[600,400], [650,400], [650,450], [600,450]])

homographyMatrix,_ = cv2.findHomography(srcPoints2, dstPoints2)

output = cv2.warpPerspective(image2, homographyMatrix, (1000,750))

cv2.imshow("Output", output)
cv2.waitKey(0)
cv2.destroyAllWindows()


#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
# Read source image.
im_src = cv2.imread('1-1.jpg')
# Four corners of the book in source image
pts_src = np.array([[467, 487],
       [794, 542],
       [356, 588],
       [761, 674]])


    
# Read destination image.
    

    
# Four corners of the book in destination image.
pts_dst = np.array([[300,100] , [300,450],[50,100],[50,450]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
    
# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (800, 600))

# Display image   
cv2.imshow("Image", im_out)



cv2.waitKey(0)

#%% Part 3

import cv2
import numpy as np

def mouse_handler(event, x, y, flags, data) :
    
    if event == cv2.EVENT_LBUTTONDOWN :
        cv2.circle(data['im'], (x,y),3, (0,0,255), 5, 16);
        cv2.imshow("Image", data['im']);
        if len(data['points']) < 4 :
            data['points'].append([x,y])

def get_four_points(im):
    
    # Set up data to send to mouse handler
    data = {}
    data['im'] = im.copy()
    data['points'] = []
    
    #Set the callback function for any mouse event
    cv2.imshow("Image",im)
    cv2.setMouseCallback("Image", mouse_handler, data)
    cv2.waitKey(0)
    
    # Convert array to np.array
    points = np.vstack(data['points']).astype(float)
    
    return points


if __name__ == '__main__' :

    # Read in the image.
    img = cv2.imread('2-1.jpg')

    # resizing image
    scale_percent = 50
    width = int(img.shape[1]*scale_percent/100)
    height = int(img.shape[0]*scale_percent/100)
    dim = (width, height)
    im_Street = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)


    #size = (400, 300, 3)
    #im_Tablo = np.zeros(size, np.uint8)
    
    
    
    # Read in the image.
    img2 = cv2.imread('2-3.jpg')

    # resizing image
    scale_percent2 = 20
    width2 = int(img2.shape[1]*scale_percent2/100)
    height2 = int(img2.shape[0]*scale_percent2/100)
    dim2 = (width2, height2)
    im_Tablo = cv2.resize(img2, dim2, interpolation = cv2.INTER_AREA)
    
     # Destination image
    size = (width2,height2,3)
    
    
    

    pts_Tablo = np.array(
                       [
                        [0,0],
                        [size[0] - 1, 0],
                        [size[0] - 1, size[1] -1],
                        [0, size[1] - 1 ]
                        ], dtype=float
                       )
    
   
   
    '''
        Click on the four corners of the book -- top left first and
        bottom left last -- and then hit ENTER
        '''
    
    # Show image and wait for 4 clicks.
    cv2.imshow("Image", im_Street)
    pts_Street = get_four_points(im_Street);
    
    # Calculate the homography
    h, status = cv2.findHomography( pts_Tablo, pts_Street)

    # Warp source image to destination
    src_warped = cv2.warpPerspective(im_Tablo, h, (im_Street.shape[1],im_Street.shape[0]))
    
    
    
        
    
    for i in range(im_Street.shape[0]):
        for j in range(im_Street.shape[1]):
            x = src_warped[i][j]
            if(x.all() != 0):
                im_Street[i][j][0] = 0
                im_Street[i][j][1] = 0
                im_Street[i][j][2] = 0

    
    
    
    # Put src_warped over dst
    result = cv2.add(im_Street, src_warped)


    # Show output
    cv2.imshow('image', result)
    cv2.imwrite("result.jpg", result)
    
    cv2.waitKey(0)
    
#%% Question 4

# Functions
import numpy as np
import cv2

def stitch(images, ratio=0.75, reprojThresh=4.0,
  showMatches=False):
  (imageB, imageA) = images
  (kpsA, featuresA) = detectAndDescribe(imageA)
  (kpsB, featuresB) = detectAndDescribe(imageB)
  M = matchKeypoints(kpsA, kpsB,
   featuresA, featuresB, ratio, reprojThresh)
  if M is None:
   return None
  (matches, H, status) = M
  result = cv2.warpPerspective(imageA, H,
   (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
  result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
  return result
    
def detectAndDescribe(image):
    
    akaze = cv2.AKAZE_create()
    (kps, features) = akaze.detectAndCompute(image, None)
    
    kps = np.float32([kp.pt for kp in kps])
    return (kps, features)

def matchKeypoints(kpsA, kpsB, featuresA, featuresB,
  ratio, reprojThresh):
  matcher = cv2.DescriptorMatcher_create("BruteForce")
  rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
  matches = []
  for m in rawMatches:
   if len(m) == 2 and m[0].distance < m[1].distance * ratio:
    matches.append((m[0].trainIdx, m[0].queryIdx))
                
  if len(matches) > 4:
   ptsA = np.float32([kpsA[i] for (_, i) in matches])
   ptsB = np.float32([kpsB[i] for (i, _) in matches])
   (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
    reprojThresh)
   return (matches, H, status)
  return None
    
def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
  (hA, wA) = imageA.shape[:2]
  (hB, wB) = imageB.shape[:2]
  vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
  vis[0:hA, 0:wA] = imageA
  vis[0:hB, wA:] = imageB
  for ((trainIdx, queryIdx), s) in zip(matches, status):
   if s == 1:
    ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
    ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
    cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
  return vis
   

#%% 

image1 = cv2.imread('3-1.jpeg') 
image2 = cv2.imread('3-2.jpeg')
             
result = stitch([image1, image2], showMatches=True)

cv2.imshow("image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
    
#%% Question 5

# Part 1 Image 1
image1 = cv2.imread('4-1.jpg')
 
    

retval1, corners1 = cv2.findChessboardCorners(image1,(7,8))


for i in range(len(corners1)):
     points1 = corners1[i]
     image1 = cv2.circle(image1,(points1[0][0],points1[0][1]),3,(255,0,0),cv2.FILLED)
     
cv2.imshow("image1left", image1)     

retval2, corners2 = cv2.findChessboardCorners(image1,(6,5))

image1 = cv2.imread('4-1.jpg')

for i in range(len(corners2)):
     points2 = corners2[i]
     image1 = cv2.circle(image1,(points2[0][0],points2[0][1]),3,(255,0,0),cv2.FILLED)


#pts1 = np.concatenate((corners1, corners2))
pts1 = np.row_stack((corners1, corners2))          
cv2.imshow("image1right", image1)       
     

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Part 1 Image 2

image2 = cv2.imread('4-2.jpg')       


retval3, corners3 = cv2.findChessboardCorners(image2,(7,8))


for i in range(len(corners3)):
     points3 = corners3[i]
     image2 = cv2.circle(image2,(points3[0][0],points3[0][1]),3,(255,0,0),cv2.FILLED)
     
cv2.imshow("image2left", image2)     

retval4, corners4 = cv2.findChessboardCorners(image2,(6,5))

image2 = cv2.imread('4-2.jpg')

for i in range(len(corners4)):
     points4 = corners4[i]
     image2 = cv2.circle(image2,(points4[0][0],points4[0][1]),3,(255,0,0),cv2.FILLED)
          
     
#pts2 = np.concatenate((corners3, corners4))

pts2 = np.row_stack((corners3, corners4))
        
cv2.imshow("image2right", image1)       
     

cv2.waitKey(0)
cv2.destroyAllWindows()


#%% Fundamental Matrix
# Part 2

# Finding Fundamental Matrix And Saving It
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

# Saving Fundamental Matrix


import numpy as np
np.save("array.npy", np.array(F))
F = np.load("array.npy")


#%% Functions Of Part 3 and 4

def drawOneline(image1,image2,r,point):
    _,c,_ = image1.shape
    color = tuple(np.random.randint(0,255,3).tolist())
    x0,y0 = map(int, [0, -r[2]/r[1] ])
    x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
    image1 = cv2.line(image1, (x0,y0), (x1,y1), color, 3, 1)
    image2 = cv2.circle(image2,tuple(point),5, color,-1)
    return image1,image2


def drawSeveralLines(image1,image2,points):
    for i in range(points.shape[0]):
        point = points[i,:,:]
        print(point)
        FoundedLine = cv2.computeCorrespondEpilines( point[0].reshape(-1,1,2), 2, F)
        image1, image2 = drawOneline(image1, image2, FoundedLine.reshape(3), point[0])
        
    return image1,image2  

#%% Part 3

image3 = cv2.imread('4-3.jpg')
image4 = cv2.imread('4-4.jpg')
 

FoundedLine = cv2.computeCorrespondEpilines( np.array([265,305]).reshape(-1,1,2), 2, F)
 
image3 , image4 = drawOneline(image3, image4, FoundedLine.reshape(3), np.array([265,305]) )

cv2.imshow("image3", image3)
cv2.imshow("image4", image4)

cv2.waitKey(0)
cv2.destroyAllWindows()



 
#%% Part 4

# Drawing pts2 and their lines

image3 = cv2.imread('4-3.jpg')
image4 = cv2.imread('4-4.jpg')



image3 , image4 = drawSeveralLines(image3, image4, pts2 )


cv2.imshow("image3", image3)
cv2.imshow("image4", image4)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%%

# Drawing pts1 and their lines

image3 = cv2.imread('4-3.jpg')
image4 = cv2.imread('4-4.jpg')

image4 , image3 = drawSeveralLines(image4, image3, pts1 )


cv2.imshow("image3", image3)
cv2.imshow("image4", image4)

cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Simultaniesly
  
image3 = cv2.imread('4-3.jpg')
image4 = cv2.imread('4-4.jpg')


image3 , image4 = drawSeveralLines(image3, image4, pts2 )
image4 , image3 = drawSeveralLines(image4, image3, pts1 )

cv2.imshow("image3", image3)
cv2.imshow("image4", image4)

cv2.waitKey(0)
cv2.destroyAllWindows()



