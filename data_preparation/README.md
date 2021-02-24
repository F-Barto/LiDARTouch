<div align="center">  

# Details on the data preparations

</div>

## KITTI

The KITTI dataset [1] is splet into several sub-set each made for a specific task (Odometry, 3D tracking, etc...)

In our project, we use:
- KITTI raw [link to page](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- KITTI Odometry [link to page](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)

Across all sub-sets the sensor setup is the same and is detailed here:
- http://www.cvlibs.net/datasets/kitti/setup.php

![](http://www.cvlibs.net/datasets/kitti/images/setup_top_view.png)

notes:
- The laser scanner spins at 10 frames per second (**10Hz**, ~100k points per cycle) and has 64 laser beams
- Image resolution is **1392 x 512 pixels**
- The cameras are triggered at **10 frames per second** by the laser scanner
- Each subset has a devkit downloadable on their respectives pages. In each devkit you can find a README explaining the format of each files.

## KITTI Raw

From the KITTI Raw dataset we use the following:
- Raw processed (synced+rectified) color stereo sequences (0.5 Megapixels, stored in png format)
- 3D Velodyne point clouds (100k points per frame, stored as binary float matrix)
- 3D GPS/IMU data (location, speed, acceleration, meta information, stored as text file)
- Calibration (Camera, Camera-to-GPS/IMU, Camera-to-Velodyne, stored as text file)

The 3D GPS/IMU data is saved in OXTS format

Note that, on KITTI, some frames have incorrect speed

## KITTI Odometry

From the KITTI Odometry dataset we use the following:
- color images 
- 3D velodyne laser data
- calibration files
- GPS/OXTS ground truth


Notes:
- Raw GPS coordinates are noisy, but poses from kitti odometry are somewhat cleaned and have a better quality.

##  References

Please cite their work if you use it. 

[1] Andreas Geiger et al., Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, CVPR 2012 (http://www.cvlibs.net/datasets/kitti/index.php)
```
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}
```

## Argoverse

Argoverse have two datasets:
1. **Argoverse Tracking**
2. **Argoverse Forecasting**

### Argoverse Tracking



### Argoverse Forecasting

csv format:

``TIMESTAMP,TRACK_ID,OBJECT_TYPE,X,Y,CITY_NAME``