# NIR Vein Finder
 
The purpose of this program is to be able to process the live image feed coming from the NIR sensor and use image segmentation to help delineate the veins in the desired area of imaging.

## Description

The development of the vein finder system's image segmentation capabilities consists of seven main sections: color space detection, tone curve, thresholding, contour analysis, morphing, drwing annotations, and tone mapping all of which working hand in hand to highlight and delineate the veins at a specific area.


## Getting Started

### Dependencies

* Python 3.12
* OpenCV
* NumPy

### Installing

* Download the ZIP file and extract to desired location. (Same if running on Raspberry Pi with NoIR camera)

### Executing program

* If testing out on personal computer or Raspberry Pi without NoIR camera, please use label.py file. Simply load an image which will be processed by typing out the image location in the filename line of the code. After that, run the program to view the changes with each image segmentation function being applied, make changes to main to display the desired function by commenting out the unwanted functions.
* If running live on a Raspberry Pi with the NoIR camera, simply run the VeinFinder.py program to view the results of the image segmentation, make changes to main to display the desired function by commenting out the unwanted functions. To close the program, press q or ctrl z to end terminate the program.

## Help

Any advise for common problems or issues.
```
If program does not work, double chack to make sure the the NoIR camera is connected, or that you are using the label.py program.
```

## Authors

Contributors names and contact info

 Lance Delos Reyes
 u1248801@utah.edu
 GitHub: lanster108

## Version History

* 0.2
    * Added supplementary files
* 0.1
    * Initial Release

## License



## Acknowledgments

Inspiration, code snippets, etc.
* https://learnopencv.com/automated-image-annotation-tool-using-opencv-python/
* https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
* https://www.analyticsvidhya.com/blog/2022/07/a-brief-study-of-image-thresholding-algorithms/#:~:text=Image%20thresholding%20is%20a%20type,is%20done%20in%20grayscale%20images. 
* https://www.cambridgeincolour.com/tutorials/levels.htm, https://www.sciencedirect.com/topics/engineering/image-histogram 
* https://www.cs.auckland.ac.nz/courses/compsci773s1c/lectures/ImageProcessing-html/topic4.htm 
* https://learnopencv.com/automated-image-annotation-tool-using-opencv-python/
* https://github.com/richbai90/img_labeler
