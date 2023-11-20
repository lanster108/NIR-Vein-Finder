import xml.etree.cElementTree as ET
import numpy as np
from os import path

# create an enum for the different bounding box formats
class BBoxFmt():
    XYXY = 0
    XYWH = 1
    XXYY = 2
    YYXX = 3
    YXYX = 4

class VOCWriter():
    def __init__(self, filename: str, image: np.ndarray, segmented: False, pretty: bool = True, encoding: str = 'utf-8', indent: str = '    '):
        '''
        Create a new VOCWriter object.
        
        Parameters
        ----------
        filename: str the path to the voc file to be created
        
        image: np.array the image to be associated with the voc file
        
        [segmented]: bool whether the image is segmented, defaults to False
        
        [pretty]: bool whether the XML should be pretty printed, defaults to True
        
        [encoding]: str the encoding to use when writing the XML, defaults to 'utf-8'
        
        [indent]: str the string to use for indentation if pretty is True, defaults to 4 spaces
        '''
        # store the formatting parameters
        self.pretty = pretty
        self.encoding = encoding
        self.indent = indent
        # Store the filename for later use
        self.__filename = filename
        # Get the image dimensions and depth
        if len(image.shape) == 3:
            w, h, d = image.shape
        else:
            w, h = image.shape
            d = 1
        # Create the root element
        self.__root = ET.Element('annotation')
        # Create the subelements that are always present
        folder = ET.SubElement(self.__root, 'folder')
        folder.text = path.basename(path.dirname(filename))
        filename = ET.SubElement(self.__root, 'filename')
        filename.text = path.basename(self.__filename)
        p = ET.SubElement(self.__root, 'path')
        p.text = self.__filename
        db = ET.SubElement(ET.SubElement(self.__root, 'source'), 'database')
        db.text = 'Unknown'
        size = ET.SubElement(self.__root, 'size')
        width = ET.SubElement(size, 'width')
        width.text = str(w)
        height = ET.SubElement(size, 'height')
        height.text = str(h)
        depth = ET.SubElement(size, 'depth')
        depth.text = str(d)
        seg = ET.SubElement(self.__root, 'segmented')
        seg.text = str(int(segmented))
        
        # Create the file object that will be used to write the XML
        # It will be created when the context is entered
        self.__file = None
        
        
    
    def __enter__(self):
        # Open the file for writing
        self.__file = open(self.__filename, 'w')
        return self
    
    def __exit__(self, type, value, traceback):
        # Write the XML to the file
        if self.pretty:
            ET.indent(self.__root, space=self.indent)
        self.__file.write(ET.tostring(self.__root, encoding=self.encoding).decode(self.encoding))
        # Close the file
        self.__file.close()

    def annotate(self, class_name: str, bbox: tuple, difficult: bool = False, truncated: bool = False, bbox_fmt: BBoxFmt = BBoxFmt.XYWH) -> None:
        '''
        Create an annotation for the current image.
        Creates an object element with the given class name and bounding box.
        
        Parameters
        ----------
        class_name: str the name of the class to annotate
        
        bbox: tuple the bounding box to annotate
        
        [difficult]: bool whether the object is difficult to detect, defaults to False
        
        [truncated]: bool whether the object is truncated, defaults to False
        
        [bbox_fmt]: BBoxFmt the format of the bounding box, defaults to BBoxFmt.XYWH
        '''
        if bbox_fmt == BBoxFmt.XYXY:
            x1, y1, x2, y2 = bbox
        elif bbox_fmt == BBoxFmt.XYWH:
            x1, y1 = bbox[0], bbox[1]
            x2, y2 = bbox[0] + bbox[2], bbox[1] + bbox[3]
        elif bbox_fmt == BBoxFmt.XXYY:
            x1, x2, y1, y2 = bbox
        elif bbox_fmt == BBoxFmt.YYXX:
            y1, y2, x1, x2 = bbox
        elif bbox_fmt == BBoxFmt.YXYX:
            y1, x1, y2, x2 = bbox
        
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        
        # Create the object element
        obj = ET.SubElement(self.__root, 'object')
        obj_name = ET.SubElement(obj, 'name')
        obj_name.text = class_name
        obj_pose = ET.SubElement(obj, 'pose')
        obj_pose.text = 'Unspecified'
        obj_truncated = ET.SubElement(obj, 'truncated')
        obj_truncated.text = str(int(truncated))
        obj_difficult = ET.SubElement(obj, 'difficult')
        obj_difficult.text = str(int(difficult))
        obj_bndbox = ET.SubElement(obj, 'bndbox')
        bndbox_xmin = ET.SubElement(obj_bndbox, 'xmin')
        bndbox_xmin.text = str(x1)
        bndbox_ymin = ET.SubElement(obj_bndbox, 'ymin')
        bndbox_ymin.text = str(y1)
        bndbox_xmax = ET.SubElement(obj_bndbox, 'xmax')
        bndbox_xmax.text = str(x2)
        bndbox_ymax = ET.SubElement(obj_bndbox, 'ymax')
        bndbox_ymax.text = str(y2)