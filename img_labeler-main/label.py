import imghdr
import matplotlib.pyplot as plt
import cv2
import numpy as np
from os import path, listdir
from voc_writer import VOCWriter, BBoxFmt
plt.rcParams['image.cmap'] = 'gray'

def read_image(filename):
    filename = "/home/lance/Desktop/img_labeler-main/apple.jpg"
    '''
    Read an image from a path. If the path is a directory, select a random image from the directory.
    
    parameters
    ----------
    filename: str the path to the image or directory
    
    returns
    -------
    img: np.array the image in BGR format
    '''
    # if the path does not include a file name, select a random image from the path
    if path.isdir(filename):
        filename = path.join(filename, np.random.choice([f for f in listdir(filename) if f.endswith('.jpg')]))
    img = cv2.imread(filename)
    
    return img

def select_colorsp(img, colorsp='gray'):
    '''
    Select a color space from an image.
    Given an image, split it into its channels and return the selected color space.
    
    Parameters
    ----------
    img: np.array the image in BGR format
    colorsp: str the color space to return. Options are 'gray', 'red', 'green', 'blue', 'hue', 'sat', 'val'
    
    Returns
    -------
    channels[colorsp]: np.array the selected color space
    '''
    # Convert to grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Split BGR.
    red, green, blue = cv2.split(img)
    # Convert to HSV.
    im_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Split HSV.
    hue, sat, val = cv2.split(im_hsv)
    # Store channels in a dict.
    channels = {'gray':gray, 'red':red, 'green':green, 
                'blue':blue, 'hue':hue, 'sat':sat, 'val':val}
     
    return channels[colorsp]

def display(im_left, im_right, name_l='Left', name_r='Right', figsize=(10,7)):
    '''
    Display two images side by side.
    
    Display two images side by side with optional titles, and optional figure size.
    
    Parameters
    ----------
    im_left: np.array the left image
    
    im_right: np.array the right image
    
    [name_l]: str the title for the left image, default is 'Left'
    
    [name_r]: str the title for the right image, default is 'Right'
    
    [figsize]: tuple the figure size, default is (10,7)
    
    Returns
    -------
    void
    ''' 
    # Flip channels for display if RGB as matplotlib requires RGB.
    im_l_dis = im_left[...,::-1]  if len(im_left.shape) > 2 else im_left
    im_r_dis = im_right[...,::-1] if len(im_right.shape) > 2 else im_right
     
    plt.figure(figsize=figsize)
    plt.subplot(121); plt.imshow(im_l_dis);
    plt.title(name_l); plt.axis(False);
    plt.subplot(122); plt.imshow(im_r_dis);
    plt.title(name_r); plt.axis(False);
    plt.show(block=True)
    
def sToneCurve(frame):
    look_up_table = np.zeros((256, 1), dtype='uint8')
    for i in range(256):
        look_up_table[i][0] = 255 * 0.1 *  (np.sin(np.pi * (i / 255 - 1 / 2)) + 1) / 2
    return cv2.LUT(frame, look_up_table)


def threshold(img, thresh=127, mode='inverse'):
    '''
    Threshold an image.
    
    Threshold an image using a given threshold value and mode.
    
    Parameters
    ----------
    img: np.array the image to threshold
    
    [thresh]: int the threshold value, default is 127
    
    [mode]: str the threshold mode, options are 'direct' and 'inverse', default is 'inverse'
    
    Returns
    -------
    The thresholded image.
    '''
    im = img.copy()
     
    if mode == 'direct':
        thresh_mode = cv2.THRESH_BINARY
    else:
        thresh_mode = cv2.THRESH_BINARY_INV
     
    _, thresh = cv2.threshold(im, thresh, 150, thresh_mode)
         
    return thresh

def get_bboxes(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Sort according to the area of contours in descending order.
    sorted_cnt = sorted(contours, key=cv2.contourArea, reverse = True)
    # Remove max area, outermost contour.
    sorted_cnt.remove(sorted_cnt[0])
    bboxes = []
    for cnt in sorted_cnt:
        x,y,w,h = cv2.boundingRect(cnt)
        cnt_area = w * h
        bboxes.append((x, y, x+w, y+h))
    return bboxes

def morph_op(img, mode='open', ksize=5, iterations=1):
    im = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize, ksize))
     
    if mode == 'open':
        morphed = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    elif mode == 'close':
        morphed = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)
    elif mode == 'erode':
        morphed = cv2.erode(im, kernel)
    else:
        morphed = cv2.dilate(im, kernel)
     
    return morphed

def draw_annotations(img, bboxes, thickness=2, color=(0,255,0)):
    annotations = img.copy()
    for box in bboxes:
        tlc = (box[0], box[1])
        brc = (box[2], box[3])
        cv2.rectangle(annotations, tlc, brc, color, thickness, cv2.LINE_AA)
     
    return annotations

def filter_bboxes_by_area(img, bboxes, min_area_ratio=0.001):
    filtered = []
    # Image area.
    im_area = img.shape[0] * img.shape[1]
    for box in bboxes:
        x,y,w,h = box
        cnt_area = w * h
        # Remove very small detections.
        if cnt_area > min_area_ratio * im_area:
            filtered.append(box)
    return filtered

def filter_bboxes_by_xy(bboxes, min_x=None, max_x=None, min_y=None, max_y=None):
    filtered_bboxes = []
    for box in bboxes:
        x, y, w, h = box
        x1, x2 = x, x+w
        y1, y2 = y, y+h
        if min_x is not None and x1 < min_x:
            continue
        if max_x is not None and x2 > max_x:
            continue
        if min_y is not None and y1 < min_y:
            continue
        if max_y is not None and y2 > max_y:
            continue
        filtered_bboxes.append(box)
    return filtered_bboxes

def save_annotations(img: np.ndarray, filename: str, bboxes: tuple, class_name:str ='0', bb_format: BBoxFmt = BBoxFmt.XYXY):
    writer = VOCWriter(filename, img, False)
    with writer as w:
        for box in bboxes:
            w.annotate(class_name, box, bbox_fmt=bb_format)
            

def main():
    # read the image
    # for f in listdir(path.join(path.dirname(__file__), 'frames')):
    #     img = read_image(path.join(path.dirname(__file__), 'frames', f))
        # mask the image:
            # Select colorspace.
            # Perform thresholding.
            # Perform morphological operations.
            # Contour analysis to find bounding boxes.
            # Filter unnecessary contours.
            # Draw annotations.
            # Save in the required format.
        img = read_image(path.join('C:/Users/User/Downloads', 'veins.jpg'))
        green_img = select_colorsp(img, colorsp='green')
        display(img, green_img, 'Original', 'Green')
        threshed_img = threshold(green_img, thresh=85)
        display(img, threshed_img, 'Original', 'Thresholded')
        morphed_img = morph_op(threshed_img, mode='open', ksize=3, iterations=3)
        bboxes = get_bboxes(morphed_img)
        filtered_bboxes = filter_bboxes_by_area(morphed_img, bboxes, min_area_ratio=0.001)
        filtered_bboxes = filter_bboxes_by_xy(filtered_bboxes, min_x=175, min_y=20)
        display(img, draw_annotations(img, filtered_bboxes, thickness=1, color=(0,255,0)))
        #save_annotations(img, path.join(path.dirname(__file__), 'labels', 'Annotations', path.basename(f).split('.')[0] + '.xml'), filtered_bboxes, class_name='fluorophore', bb_format=BBoxFmt.XYXY)
        image_contrasted = sToneCurve(green_img)
        display(green_img, image_contrasted)
        threshed_img2 = threshold(image_contrasted, thresh=5)
        display(image_contrasted, threshed_img2, 'Original', 'Thresholded')
    
def countTonemap(hdr, min_fraction=0.0005):
	counts, ranges = np.histogram(hdr, 256)
	min_count = min_fraction * hdr.size
	delta_range = ranges[1] - ranges[0]

	img = hdr.copy()
	for i in range(len(counts)):
            if counts[i] < min_count:
                img[img >= ranges[i + 1]] -= delta_range
                ranges -= delta_range
                return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)



if __name__ == '__main__':
    main()
