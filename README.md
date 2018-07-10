## Reference 
- TensorFlow Tutorial For Beginners
    - https://www.datacamp.com/community/tutorials/tensorflow-tutorial
    - https://github.com/datacamp/datacamp-community-tutorials

- Traffic Sign Recognition with Tensorflow
    - https://www.jianshu.com/p/d8feaddc7bdf
    - https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb

- TensorFlow从基础到实战：一步步教你创建交通标志分类神经网络
    https://www.jiqizhixin.com/articles/2017-07-30-3  

## Setup
```
$ virtualenv --system-site-packages -p python3 venv3

$ . venv3/bin/activate

(venv3)$ pip3 install -r requirements.txt

...
(venv3)$ deactivate
```

## Data
    http://btsd.ethz.ch/shareddata/
    BelgiumTS for Classification (cropped images)
    The testing and training data folder contain 62 subfolders, which are the 62 types of traffic signs that you’ll use for classification

    The data directory contains sub-directories with sequental numerical names from 00000 to 00061. 
    The name of the directory represents the labels from 0 to 61, and the images in each directory represent the traffic signs that belong to that label. 
    The images are saved in the not-so-common .ppm format, skimage library can handle it.