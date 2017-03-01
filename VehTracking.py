''' Udacity Self-Driving Car Engineer Nanodegree
    Project 5
    Vehicle Detection and Tracking '''

import numpy as np
import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import glob
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from moviepy.editor import VideoFileClip
from moviepy.editor import ImageSequenceClip
from moviepy.editor import ImageClip
from moviepy.editor import VideoClip
from scipy.ndimage.measurements import label

#--------------#
### Clean up ###
#--------------#
try:
    os.remove('tmp_images/.DS_Store')
except:
    pass

#---------------------------------#
### Get feature vector function ###
#---------------------------------#

def get_feature(img, spatial_size = (32, 32), hist_width = 4, n_bins = 32, orient = 12, pix = 8, cells = 2):
    # take in an RGB image
    # extract color histograms
    # extract spatial color data
    # extract hog data
    # concatenate

    # metadata
    height = img.shape[0]

    # feature maximum value for normalizing
    feat_max = np.float32(img.shape[0]*img.shape[1])

    # hog data
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_array = hog(img_gray, orientations=orient, pixels_per_cell=(pix, pix),
                    cells_per_block=(cells, cells), visualise=False, feature_vector=True)
    hog_array = feat_max*hog_array/max(hog_array)

    # color histograms
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # color histograms
    img_S = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]
    rhist = np.histogram(img[:,:,0], bins=n_bins, range=(0, 256))
    ghist = np.histogram(img[:,:,1], bins=n_bins, range=(0, 256))
    bhist = np.histogram(img[:,:,2], bins=n_bins, range=(0, 256))
    shist = np.histogram(img_S[:,:], bins=n_bins, range=(0, 256))

    # spatial color data
    small_rgb = cv2.resize(img, (spatial_size[0], spatial_size[1]))
    spatial_rgb = np.array(small_rgb.ravel(),dtype = np.float)
    spatial_rgb *= feat_max / 255.0
    small_s = cv2.resize(img_S, (spatial_size[0], spatial_size[1]))
    spatial_s = np.array(small_s.ravel(),dtype = np.float)
    spatial_s *= feat_max / 255.0


    feature = np.concatenate((hog_array, rhist[0], ghist[0], bhist[0], shist[0], spatial_rgb, spatial_s))/feat_max
    feature = feature.astype(np.float)-0.5

    return feature

#----------------------------#
### Class for bounding box ###
#----------------------------#

class BBox():
    # Bounding box to search in an image
    def __init__(self, size, stride, origin, stop):
        # coordinates of the upper left corner to start search
        self.origin = origin
        self.origindeep = origin #never lose the original origin
        # width and height
        self.size = size
        # horizontal and vertical stride for scanning
        self.stride = stride
        # coordinates of the upper left corner of the box
        self.start = self.origin
        # y coordinate above which to scan
        self.stop = stop
        # flag to keep track of box within boundary
        self.out_of_bounds = 0

    def new_origin(self, origin):
        # update the origin of the box
        self.origin = origin
        self.start = origin

    def end(self):
        # update the end (lower-right corner) of the box
        x = self.start[0]+self.size[0]
        y = self.start[1]+self.size[1]
        return (x, y)

    def walk_right(self):
        # move the box in the x direction
        x = self.start[0] + self.stride[0]
        y = self.start[1]
        self.start = (x, y)
        self.end()

    def walk_down(self):
        # move the box in the y direction, reset x
        x = 0
        y = self.start[1] + self.stride[1]
        self.start = (x, y)
        self.end()

    def __str__(self):
        return "Start = %s, End = %s" % (self.start, self.end())

    def reset(self):
        # set the origin back to its original place
        self.origin = self.origindeep
        self.start = self.origin

#---------------------------------------------#
### Class for list of boxes containing cars ###
#---------------------------------------------#
class BoxList():
    # Bounding box to search in an image
    def __init__(self):
        self.list = []

    def append(self, box):
        self.list.append([box.start[0], box.start[1], box.end()[0], box.end()[1]])

    def reset(self):
        self.list = []

#------------------------#
### Class for heat map ###
#------------------------#
class HeatMap():
    #blank images
    def __init__(self, image):
        self.img = np.zeros_like(image)
        self.cool_rate = 25
        self.threshold = 30

    def addheat(self, boxlist):
        for idx in range(len(boxlist.list)):
            x1 = boxlist.list[idx][0]
            y1 = boxlist.list[idx][1]
            x2 = boxlist.list[idx][2]
            y2 = boxlist.list[idx][3]
            for x in range(x1, x2):
                for y in range(y1, y2):
                    if self.img[y, x, 0] < 245:
                        self.img[y, x, 0] += 10
                    
    def cool(self):
        self.img[self.img < self.cool_rate] = 0
        self.img[self.img >= self.cool_rate] -= self.cool_rate

    def thresh(self):
        img_thresh = self.img
        img_thresh[img_thresh <= self.threshold] = 0

        return img_thresh

    def reset(self):
        self.img = 0

#----------------#
### Draw boxes ###
#----------------#

def draw_box(img, box, boxlist):
    # draw a box on an image
    img_copy = np.copy(img)

    # check if box fits in x direction
    if box.end()[0] <= img.shape[1]:
        # check if box fits in y direction
        if box.end()[1] <= box.stop:
            cv2.rectangle(img_copy, box.start, box.end(), (0,0,255), 6)
            small_area = img[box.start[1]:box.end()[1], box.start[0]:box.end()[0]]
            small_img = cv2.resize(small_area, (64, 64))
            feat = get_feature(small_img)
            if svc.predict(feat) == 1:
                cv2.rectangle(img, box.start, box.end(), (255,0,0), 6)
                boxlist.append(box)
            
            box.walk_right()

        # if box fits in x but not y, out of bounds
        else:
            box.out_of_bounds = 1

    # if box does not fit in x, walk down
    else:
        box.walk_down()

        # check if box fits in y direction
        if box.end()[1] <= box.stop:
            cv2.rectangle(img_copy, box.start, box.end(), (0,0,255), 6)
            small_area = img[box.start[1]:box.end()[1], box.start[0]:box.end()[0]]
            small_img = cv2.resize(small_area, (64, 64))
            feat = get_feature(small_img)
            if svc.predict(feat) == 1:
                cv2.rectangle(img, box.start, box.end(), (255,0,0), 6)
                boxlist.append(box)
                                
            box.walk_right()

        # if box fits in x but not y, out of bounds
        else:
            box.out_of_bounds = 1

    return img_copy

def check_box(img, box):
    # draw a box if a car is detected
    
    # check if box fits in x direction
    if box.end()[0] <= img.shape[1]:
        # check if box fits in y direction
        if box.end()[1] <= box.stop:
            small_area = img[box.start[1]:box.end()[1], box.start[0]:box.end()[0]]
            small_img = cv2.resize(small_area, (64, 64))
            feat = get_feature(small_img)
            if svc.predict(feat) == 1:
                cv2.rectangle(small_img, (2,2),(62, 62),(255,0,0),3)
            box.walk_right()

        # if box fits in x but not y, out of bounds
        else:
            box.out_of_bounds = 1

    # if box does not fit in x, walk down
    else:
        box.walk_down()

        # check if box fits in y direction
        if box.end()[1] <= box.stop:
            small_area = img[box.start[1]:box.end()[1], box.start[0]:box.end()[0]]
            small_img = cv2.resize(small_area, (64, 64))
            feat = get_feature(small_img)
            if svc.predict(feat) == 1:
                cv2.rectangle(small_img, (2,2),(62, 62),(255,0,0),3)
            box.walk_right()

        # if box fits in x but not y, out of bounds
        else:
            box.out_of_bounds = 1
            small_img = []

    return small_img

def make_heat_map_image(image):
    
    img_copy = np.copy(image)

    box1 = BBox(size = (196,196), stride = (32, 32), origin = (0,350), stop = 680)
    box2 = BBox(size = (128,128), stride = (24, 24), origin = (0,350), stop = 680)
    box3 = BBox(size = (96,96), stride = (16, 16), origin = (0,350), stop = 550)

    boxes = [box1, box2, box3]
    boxlist = BoxList()
    heatmap.cool()

    for box in boxes:
        while box.out_of_bounds == 0:
            _ = draw_box(img_copy, box, boxlist)

    heatmap.addheat(boxlist)
    image_thresh = heatmap.thresh()
    labels = label(image_thresh)
    for box_label in range(labels[1]):
        pixels = (labels[0] == (box_label+1)).nonzero()
        x1 = np.min(np.array(pixels[1]))
        x2 = np.max(np.array(pixels[1]))
        y1 = np.min(np.array(pixels[0]))
        y2 = np.max(np.array(pixels[0]))
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,255,0), 6)

    return image

#-----------------------------------#
### Create feature and label sets ###
#-----------------------------------#

def get_filepaths(dirname):
    # get list of all files in a directory
    # http://stackoverflow.com/questions/3207219/how-to-list-all-files-of-a-directory
    flist = []

    for (dirpath, dirnames, filenames) in os.walk(dirname):
        for file in filenames:
            filepath = os.path.join(dirpath, file)
            flist.append(filepath)

    return flist

vehpathlist = get_filepaths('vehicles')
nvehpathlist = get_filepaths('non-vehicles')

vehlist = [] # list of vehicle images
nvehlist = [] # list of non-vehicle images
for path in vehpathlist:
   if path.endswith(".png"):
       vehlist.append(path)

for path in nvehpathlist:
    if path.endswith(".png"):
        nvehlist.append(path)

num_veh_imgs = len(vehlist)
num_nveh_imgs = len(nvehlist)
num_imgs = num_veh_imgs + num_nveh_imgs

# create an array of feature filenames
feature_filenames = vehlist+nvehlist
train_veh_filenames = vehlist[0:int(num_veh_imgs*0.8)]
test_veh_filenames = vehlist[int(num_veh_imgs*0.8):num_veh_imgs]
train_nveh_filenames = nvehlist[0:int(num_nveh_imgs*0.8)]
test_nveh_filenames = nvehlist[int(num_nveh_imgs*0.8):num_nveh_imgs]
train_filenames = train_veh_filenames+train_nveh_filenames
test_filenames = test_veh_filenames+test_nveh_filenames

# create an array of labels
labels = np.append(np.ones((num_veh_imgs,1),dtype = np.uint8),
                   np.zeros((num_nveh_imgs,1), dtype = np.uint8))
y_train = np.append(np.ones((len(train_veh_filenames),1),dtype = np.uint8),
                   np.zeros((len(train_nveh_filenames),1), dtype = np.uint8))
y_test = np.append(np.ones((len(test_veh_filenames),1),dtype = np.uint8),
                   np.zeros((len(test_nveh_filenames),1), dtype = np.uint8))

num_train = (len(train_filenames))
num_test = (len(test_filenames))

#-----------------------------#
### Get feature information ###
#-----------------------------#
img = mpimg.imread(feature_filenames[0])

# size of image space over which to extract features
feat_H, feat_W = img.shape[0], img.shape[1]

# length of feature vector after extraction
feat_L = len(get_feature(img))

#-----------------------------------#
### Create feature from one image ###
#-----------------------------------#

def get_image(fname):
    # get an image in png format, put in RGB format
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

img = get_image(feature_filenames[0])
plt.subplot(2,4,1), plt.imshow(img), plt.title('Original'), plt.xticks([0, 16, 32, 48, 64])

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

feature_array, hog_img, = hog(img_gray, orientations=8, pixels_per_cell=(8, 8),
                    cells_per_block=(2, 2), visualise=True, feature_vector=True)

plt.subplot(2,4,2), plt.imshow(hog_img, 'gray'), plt.title('Hog'), plt.xticks([0, 16, 32, 48, 64])

img_S = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[:,:,1]
plt.subplot(2,4,3), plt.imshow(img_S, 'gray'), plt.title('Saturation'),  plt.xticks([0, 16, 32, 48, 64])

rhist = np.histogram(img[:,:,0], bins=32, range=(0, 256))
ghist = np.histogram(img[:,:,1], bins=32, range=(0, 256))
bhist = np.histogram(img[:,:,2], bins=32, range=(0, 256))
shist = np.histogram(img_S[:,:], bins=32, range=(0, 256))

# Generating bin centers
bin_edges = rhist[1]
bin_centers = (bin_edges[1:]  + bin_edges[0:len(bin_edges)-1])/2

# Plot a figure with all three bar charts
plt.subplot(2,4,5)
plt.bar(bin_centers, rhist[0]), plt.xticks([0, 100, 200])
plt.xlim(0, 256)
plt.title('R Histogram')
plt.subplot(2,4,6)
plt.bar(bin_centers, ghist[0]), plt.xticks([0, 100, 200])
plt.xlim(0, 256)
plt.title('G Histogram')
plt.subplot(2,4,7)
plt.bar(bin_centers, bhist[0]), plt.xticks([0, 100, 200])
plt.xlim(0, 256)
plt.title('B Histogram')
plt.subplot(2,4,4)
plt.bar(bin_centers, shist[0]), plt.xticks([0, 100, 200])
plt.xlim(0, 256)
plt.title('S Histogram')

#-----------------------------#
### Get feature vector test ###
#-----------------------------#
a = get_feature(img)
plt.subplot(2,4,8), plt.plot(a), plt.xticks([0, 3000, 6000])
plt.title('Feature Array')
plt.tight_layout()
plt.savefig('output_images/DataExample.png',format='png')
plt.close()

#----------------------------#
### Create features vector ###
#----------------------------#
load_data = True

if load_data is False:
    X_train = np.zeros((num_train,feat_L), dtype = np.float)
    X_test = np.zeros((num_test,feat_L), dtype = np.float)
    print('Generate X_train and X_test from filenames')

    print('X_train')
    for idx in range(num_train):
        fname = train_filenames[idx]
        img = get_image(fname)
        feat = get_feature(img)
        assert len(feat) == feat_L, 'length of feature is wrong!'
        X_train[idx] = feat

    print('X_test')
    for idx in range(num_test):
        fname = test_filenames[idx]
        img = get_image(fname)
        feat = get_feature(img)
        assert len(feat) == feat_L, 'length of feature is wrong!'
        X_test[idx] = feat

    print('Save X_train.npy numpy data')
    np.save('X_train.npy', X_train, allow_pickle=True, fix_imports=True)
    print('Save X_test.npy numpy data')
    np.save('X_test.npy', X_test, allow_pickle=True, fix_imports=True)
        
else:
    print('Load X_train.npy numpy data')
    X_train = np.load('X_train.npy')
    print('Load X_test.npy numpy data')
    X_test = np.load('X_test.npy')

#--------------------------#
### Create labels vector ###
#--------------------------#

if load_data is False:
    print('Save y_train.npy numpy data')
    np.save('y_train.npy', y_train, allow_pickle=True, fix_imports=True)
    print('Save y_test.npy numpy data')
    np.save('y_test.npy', y_test, allow_pickle=True, fix_imports=True)
        
else:
    print('Load y_train.npy numpy data')
    y_train = np.load('y_train.npy')
    print('Load y_test.npy numpy data')
    y_test = np.load('y_test.npy')

#-------------------------------------------------------#
### Split the data into test/validation set and train ###
#-------------------------------------------------------#
Xtype = str(type(X_train))
Xshape = str(X_train.shape)
print('X_train is a '+Xtype+' of shape '+Xshape)
ytype = str(type(y_train))
yshape = str(y_train.shape)
print('y_train is a '+ytype+' of shape '+yshape)
Xtype = str(type(X_test))
Xshape = str(X_test.shape)
print('X_test is a '+Xtype+' of shape '+Xshape)
ytype = str(type(y_test))
yshape = str(y_test.shape)
print('y_test is a '+ytype+' of shape '+yshape)

# Use a linear SVC (support vector classifier)
iterations = 20
iter_step = 5
svc = svm.LinearSVC(max_iter = iterations, random_state = 0)
# Train the SVC
train_error_verbose = False
if train_error_verbose is True:
    train_error = np.zeros((int(iterations/iter_step),2), dtype = np.float)
    train_xaxis = np.arange(iter_step,iterations+iter_step,iter_step)
    for epoch in range(int(iterations/iter_step)):
        svc.set_params(max_iter = (epoch+1)*iter_step)
        svc.fit(X_train, y_train)
        train_error[epoch] = [svc.score(X_train, y_train),svc.score(X_test, y_test)]
        print('Run '+str(epoch+1)+': '+str((epoch+1)*iter_step)+' iterations: '+str(train_error[epoch]))

    plt.plot(train_xaxis, train_error*100)
    plt.title('Test Accuracy')
    plt.xlabel('Training iteration')
    plt.ylabel('Accuracy [%]')
    plt.legend(['Training set','Testing set'])
    plt.savefig('output_images/TrainingError.png',format='png')
    plt.close()

else:
    svc.fit(X_train, y_train)
    train_error = svc.score(X_test, y_test)
    print('Run '+str(1)+': '+str(iterations)+' iterations: '+str(train_error))

#----------------------#
### Get a test image ###
#----------------------#

clip = VideoFileClip("project_video.mp4")
test_image = clip.get_frame(t=38)

box1 = BBox(size = (196,196), stride = (32, 32), origin = (0,350), stop = 680)
box2 = BBox(size = (128,128), stride = (24, 24), origin = (0,350), stop = 680)
box3 = BBox(size = (96,96), stride = (16, 16), origin = (0,350), stop = 550)

boxes = [box1, box2, box3]
boxlist = BoxList()

img = test_image

make_scan_video = False
if make_scan_video is True:
    idx = 0
    fname = 'tmp_images/tmp_image_'+str(idx).zfill(4)+'.jpg'
    img = cv2.cvtColor(test_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname, img)
    for box in boxes:
        while box.out_of_bounds == 0:
            idx += 1
            fname = 'tmp_images/tmp_image_'+str(idx).zfill(4)+'.jpg'
            img = draw_box(test_image, box, boxlist)
            if box.out_of_bounds == 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fname, img)
    
        box.reset()

    clip = ImageSequenceClip("tmp_images", fps = 40)
    clip.write_videofile("boxscan.mp4")

make_crop_video = False
if make_crop_video is True:
    idx = 0
    for box in boxes:
        while box.out_of_bounds == 0:
            idx += 1
            fname = 'tmp_images/tmp_image_'+str(idx).zfill(4)+'.jpg'
            img = check_box(test_image, box)
            if box.out_of_bounds == 0:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(fname, img)
    
        box.reset()

    clip = ImageSequenceClip("tmp_images", fps = 40)
    clip.write_videofile("scanarea.mp4")

make_heat_video = True
if make_heat_video is True:
    heatmap = HeatMap(image = test_image)
    video_output = 'heatmap_video_out.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    clip1 = clip1.subclip(33,35)
    video_clip = clip1.fl_image(make_heat_map_image)
    video_clip.write_videofile(video_output, audio=False)


