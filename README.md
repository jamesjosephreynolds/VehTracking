# Vehicle Detection and Tracking #
The objective of this project is to develop a pipeline that takes a picture of a roadway and identifies the vehicles that are present therein.  In order to reduce the occurrence rate of false positives, information is kept between sequential images.  Single identifications are probably going to be false positives, as vehicles do not appear and disappear within a single frame of video.

## Feature Creation ##
Using the provided image set, there are 8792 images of vehicles (label = 1), and 8968 images of non-vehicles (label = 0).

In order to create a feature array for each image, I augmented data arrays of the following: HOG vector (8x8 pixels, 2x2 cells, 12 orientations), RGB red histogram (32 bins), RGB green histogram (32 bins), RGB blue histogram (32 bins), HLS saturation histogram (32 bins), RGB red spatial data (resized 32x32), RGB green spatial data (resized 32x32), RGB blue spatial data (resized 32x32) and HLS spatial data (resized 32x32).  This results in each 64x64 pixel input image having 6576 features (7\*7\*2\*2\*12+32\*4 + 1024\*4).  I applied manual normalization so that each element is scaled in the range [-0.5,0.5].

```python
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


    feature = 255*np.concatenate((hog_array, rhist[0], ghist[0], bhist[0], shist[0], spatial_rgb, spatial_s))/feat_max
    feature = feature.astype(np.uint8)

    return feature
 ```
 
The image below shows an example image, its tranformations, and the resulting feature vector.

![Whoops, where's my image](output_images/DataExample.png)
 
## Training and Validation ##
I used a LinearSVM to train my classifier.  The image below shows the test accuracy and training accuracy versus the number of training iterations.  Since overfitting appears to become an issue after around 20 iterations, that is what I chose for my final fit.

![Whoops, where's my image](output_images/TrainingError.png)

I made two errors when it came to creating training and validation datasets.  Initially I used train_test_split() to create my training and validation datasets.  This resulted in very high training and validation accuracies, but very poor performance on test images.  This implied the model was overfit.  Reviewing comments from the Confluence message boards, there is a very good point raised on this topic.  Essentially, train_test_split() integrates a shuffling step of the data.  Since the data is a set of images that are temporally very similar, shuffling these images puts nearly identical samples in the training and validation sets.  This results in extreme overfitting.

So, to overcome this issue, I implemented a blunt training/validation data split: keep the images in order and just take the last 20% for my test data.  This was a very effective suggestion by both [Mikel](https://carnd-forums.udacity.com/questions/users?username=anokas) and [Gilad](https://carnd-forums.udacity.com/questions/users?username=giladgressel).  In this way, very similar images are kept together as either training or testing information, but don't cross over.  However, my first implementation of this split resulted in 100% of the validation data being "non-vehicle".  Below is my original implementation, second implementation, and final implementation.

```python
# Original data split technique
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Second data split technique (validation data biased to non-vehicle)
X_train = features[0:14208]
X_test = features[14208:17760]
y_train = labels[0:14208]
y_test = labels[14208:17760]

# Final data split technique (training and validation sets evenly biased)
# create an array of feature filenames
X_train_veh_filenames = vehlist[0:int(num_veh_imgs*0.8)]
X_test_veh_filenames = vehlist[int(num_veh_imgs*0.8):num_veh_imgs]
X_train_nveh_filenames = nvehlist[0:int(num_nveh_imgs*0.8)]
X_test_nveh_filenames = nvehlist[int(num_nveh_imgs*0.8):num_nveh_imgs]
X_train_filenames = X_train_veh_filenames + X_train_nveh_filenames
X_test_filenames = X_test_veh_filenames + X_test_nveh_filenames

# create an array of labels
y_train = np.append(np.ones((len(train_veh_filenames),1),dtype = np.uint8),
                   np.zeros((len(train_nveh_filenames),1), dtype = np.uint8))
y_test = np.append(np.ones((len(test_veh_filenames),1),dtype = np.uint8),
                   np.zeros((len(test_nveh_filenames),1), dtype = np.uint8))
                   
X_train = np.zeros((num_train,feat_L), dtype = np.float)
X_test = np.zeros((num_test,feat_L), dtype = np.float)
for idx in range(num_train):
    fname = X_train_filenames[idx]
    img = get_image(fname)
    feat = get_feature(img)
    assert len(feat) == feat_L, 'length of feature is wrong!'
    X_train[idx] = feat

for idx in range(num_test):
    fname = X_test_filenames[idx]
    img = get_image(fname)
    feat = get_feature(img)
    assert len(feat) == feat_L, 'length of feature is wrong!'
    X_test[idx] = feat
```

## Scanning a Single Frame ##

In order to create a function to scan a frame of video, I created `class BBox()`.  Within this class are methods to update the box location within an image.

```python
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
```

Then I created a list of `BBox()` instances that I could use to iterate over a single frame.  For all bounding boxes, I began searching at y-coordinate 350.  There will not be any vehicles above this point, as it corresponds to the sky in the images.  For the smallest bounding box, I stopped the search at y-coordinate 550.  Vehicles that are below that coordinate will be close to the camera, and appear much larger than 96x96 pixels.

```python
box1 = BBox(size = (196,196), stride = (32, 32), origin = (0,350), stop = 720)
box2 = BBox(size = (128,128), stride = (24, 24), origin = (0,350), stop = 720)
box3 = BBox(size = (96,96), stride = (16, 16), origin = (0,350), stop = 550)

boxes = [box1, box2, box3]
```

The resulting scan areas can be seen in the video below.  I chose this frame for my test since it has two cars present.

[![Whoops, there should be a picture here!](https://img.youtube.com/vi/fZdrbdSeQmo/0.jpg)](https://youtu.be/fZdrbdSeQmo)

