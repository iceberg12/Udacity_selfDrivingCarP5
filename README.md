## Udacity Self-driving Car Nanodegree

---

### **Vehicle Detection and Tracking**

This Project is the fifth task of the Udacity Self-Driving Car Nanodegree program. The main goal of the project is to a software pipeline to identify vehicles in a video from a front-facing camera on a car. This goal is walked through with the following steps:

* Step 1: Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images. Also apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector.
* Step 2: Train a classifier Linear SVM classifier
* Step 3: Implement a sliding-window technique and use a trained SVM classifier to search for vehicles in images.
* Step 4: Run the pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles. Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[LUV]: ./output_images/LUV.png
[multi_detect]: ./output_images/multi_detect.png
[time_series]: ./output_images/time_series.png

---

First, we started by reading in all the `vehicle` and `non-vehicle` images. Labeled images were taken from the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples to train your classifier.  These example images come from a combination of the [GTI vehicle image database](http://www.gti.ssr.upm.es/data/Vehicle_database.html), the [KITTI vision benchmark suite](http://www.cvlibs.net/datasets/kitti/), and some examples extracted from the project video itself. All contribute to the total 8792 car and 8968 environment images in 64x64-pixel resolution.

**Note**: If you want to make the classifier works well, please consider to augment the images, at least by flipping them. This works really well for me.


### Step 1. Perform feature extraction using Histogram of Oriented Gradients (HOG), color transformation and color histogram  on a labeled training set of images.

To explore HOG features, I tried various combinations of color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  The well-performed and most simple HOG feature extraction I can find use the L channel of `LUV` color space, and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. Random images from each of the two classes are displayed here to get a feel for what the `skimage.hog()` output looks like.

![alt text][LUV]

Although HOG features are great, calculating histogram of colors and spatial color binning is an additional way to increase the detection accuracy, just like how we human detect cars on the road based on moving colors. Thus, beside `get_hog_features`, the functions `bin_spatial` with spatial bin dimension (16, 16) and `color_hist` with histogram bins 32 are implemented. The function `extract_features` combines all these features; convert from BGR if you read images through `cv2.imread`, or RGB if `mpimg.imread` is employed.

Stacking the feature vectors from these three methods give us 2432 features. This would be more if you are going to extract HOG features on all three channels, but for me I only use L-channel of LUV to give my SVM classfier an easy life (U and V channels are not representative anyway). Last but not least, remember to standardise the feature space (I use `StandardScaler` from sk-learn) before modelling to treat these features equal in magnitude.


### Step 2: Train Linear SVM Classifier

The car and no-car data is shuffled, then split into train and test data with the ratio 80 / 20.

I used GridSearchCV to look out for a good parameter `C` of the Linear SVM classifer (sk-learn). The optimal parameter is `C = 0.001`. Test accuracy is around 0.9852, so it is ok for me to move forward. However, I believe a ConvNet can do a better job. At least that's what I will try next after this project. It will be interesting to see how much ConvNet can improve over SVM, because SVM gives me some false positive in car detection, which I will show you next.


### Step 3: Sliding Window Search

Here are a few ways I learned from watching Udacity lectures, discussing on the Udacity forum and talking to my mentor:

1. Cars don't fly on the sky, so searching on the lower horizon would reduce the search time.
2. Window search size and overlap should be wisely selected. The sizes I choose are 64 and 128 pixel (it's a multi-scale searching). For 64-pixel windows I search at height range [400, 520] for far-way cars, and for 128-pixel windows I search at height range [400, 640] for nearer cars. Overlap should be as high as you can afford, but I keep it around 0.8 (meaning, ~ 52 pixels for 64-pixel windows) to make my searching below 1 second per image.

The left-hand column shows the detected windows overlaid on top of the original image.

![alt text][multi_detect]

About the code, the utility function `slide_window` defines a list of small sliding windows across a pre-defined region, while `search_windows` iterates over those windows, performs feature extraction and use the trained SVM to detect if a car is present inside that window.

Now if we review the result above, there are windows that falsely detected as cars (Test Image 5). However, we notice that there is only a single window detected. Cars in this images are still detected with at least two windows. This provides us a way to know more certainly if there is a car. By using a heatmap which increase the value of each pixel by 1 if it is covered by one window, we can see cars are seen clearer (right-hand column). During video processing, this technique can be employed further by taking a heatmap across a few frames.


### Step 4: Video Implementation

#### 1. Filter for false positives and a method for combining overlapping bounding boxes.

I recorded the positions of positive detections as heatmap in each frame of the video, called `heat_series`.  For each new frame, I take a weighted sum of these `heat_series` then thresholded that summed heatmap to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap, assuming each blob corresponded to a vehicle. Bounding boxes to cover the area of each blob is constructed by `draw_labeled_bboxes`.  

Here's an example result showing the heatmap from a series of frames of video.

![alt text][time_series]

#### 2. Performance

Here's a [link to my video result](./output_images/project_video_processed.mp4)



---

### Discussion

1. The SVM classifier gives me some false positive, so I hope I can solve this issue with a ConvNet in my next try.
2. The algorithm may have some problems in case of overlapped cars. To resolve this problem one may introduce long term memory of car centers using centroid of detected windows to track how many cars in the frame and where they are.

