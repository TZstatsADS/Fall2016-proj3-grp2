# draw the hog feature
import skimage
from skimage.feature import hog
from skimage import data, color, exposure
import matplotlib.pyplot as plt

dta = skimage.io.imread('/Users/pengfeiwang/Desktop/dogkfc/Fall2016-proj3-grp2/figs/sample_chicken.jpg')
image = color.rgb2gray(dta)

fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True)

hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.show()


# draw the haldsies
## @author: simplecv
## env: simplecv
def halfsies(left,right): 
    result = left
    crop   = right.crop(right.width/2.0,0,right.width/2.0,right.height)
    result = result.blit(crop,(left.width/2,0))
    return result

img = Image('/home/pf/ubuntu/Desktop/sample_chicken.jpg')
output = img.edges(t1=160)
result = halfsies(img,output)
result.show()
result.save('half_chicken.png')


# draw the sift feature

