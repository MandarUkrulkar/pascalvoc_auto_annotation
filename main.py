# without bbox criteria

# Import the necessary libraries
import cv2
import os
from ultralytics import YOLO
from xml.etree.ElementTree import Element, SubElement, tostring
from xml.dom.minidom import parseString
from typing import List
from functools import reduce

# Function to create XML annotations from detections
def create_xml_annotation(image_size, boxes, labels,confidence , filename, folder_name):
    annotation = Element("annotation")
    folder_element = SubElement(annotation, "folder")
    filename_element = SubElement(annotation, "filename")
    source_element = SubElement(annotation, "source")
    database_element = SubElement(source_element, "database")
    annotation_element = SubElement(source_element, "annotation")
    image_element = SubElement(source_element, "image")
    size_element = SubElement(annotation, "size")
    width_element = SubElement(size_element, "width")
    height_element = SubElement(size_element, "height")
    segmented_element = SubElement(annotation, "segmented")
    
    folder_element.text = folder_name
    filename_element.text = filename
    database_element.text = "Unknown"
    annotation_element.text = "Unknown"
    image_element.text = "Unknown"
    width_element.text = str(image_size[0])
    height_element.text = str(image_size[1])
    segmented_element.text = "0"

    for box, label, conf in zip(boxes, labels,confidence):
        xmin, ymin, xmax, ymax = box
        obj_element = SubElement(annotation, "object")
        name_element = SubElement(obj_element, "name")
        bndbox_element = SubElement(obj_element, "bndbox")
        xmin_element = SubElement(bndbox_element, "xmin")
        ymin_element = SubElement(bndbox_element, "ymin")
        xmax_element = SubElement(bndbox_element, "xmax")
        ymax_element = SubElement(bndbox_element, "ymax")
        conf_element = SubElement(obj_element, "conf")

        name_element.text = label
        xmin_element.text = str(int(xmin))
        ymin_element.text = str(int(ymin))
        xmax_element.text = str(int(xmax))
        ymax_element.text = str(int(ymax))
        conf_element.text = str(conf)
    xml_string = tostring(annotation)
    dom = parseString(xml_string)

    return dom.toprettyxml()

# Function to predict and save annotations without considering overlapping criteria
def predict_and_save_annotations(directory_path, model_path, output_directory, min_detections=1, confidence_threshold=0.4):
    # Initialize the YOLO model
    model = YOLO(model_path)

    img_dict = model.names 
    print("Class mapping from model:", img_dict)
    #img_dict = {0:'unclassified',1:'basophils',2: 'eosinophils',3: 'lymphocytes',4: 'monocytes',5: 'neutrophils'}
    #img_dict = {0:'Lymphoblast',1:'bands',2: 'basophils',3: 'blasts',4: 'eosinophils',5: 'lymphocytes',6: 'metamyelocytes',7: 'monocytes',8: 'myelocytes',9: 'neutrophils',10: 'promyelocytes',11: 'unclassified'}
    
    folder_name = os.path.basename(directory_path)

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            detections = []

            # Perform inference
            image_results = model(image, conf=confidence_threshold,iou=0.2, agnostic_nms=True)

            for bbox in image_results[0].boxes.data.tolist():
                x1, y1, x2, y2, confidence, label = bbox

                # Add all detected bounding boxes to the results without checking for overlapping
                detections.append((x1, y1, x2, y2, img_dict[int(label)],float(confidence)))

            if len(detections) >= min_detections:
                #output_image_path = os.path.join(output_directory, f"{filename[:-4]}.jpg")
                #cv2.imwrite(output_image_path, image)

                xml_annotation = create_xml_annotation(
                    image.shape[:2], 
                    [box[:4] for box in detections], 
                    [box[4] for box in detections],
                    [box[5] for box in detections], 
                    filename, 
                    folder_name
                )

                with open(os.path.join(output_directory, f"{filename[:-4]}.xml"), "w") as xml_file:
                    xml_file.write(xml_annotation)

if __name__ == "__main__":
    model_weights_path =  inpur('path for yolo model') #'/home/gpu1/Documents/Projects/Mandar/Project-WBC/comp-vis-code/multiclass_classification/ultralytics/runs/detect/train_5c_jarvis3840_yolov11l_12jan2025/weights/best.pt' 
    images_directory_path = input("Enter the directory path containing images: ")
    output_directory = input("Enter the output directory path: ")
    os.makedirs(output_directory, exist_ok=True)
    min_detections = int(input("Enter the minimum number of detections required in an image: "))
    confidence_threshold = float(input("Enter the confidence threshold for detections (0-1): "))
    predict_and_save_annotations(images_directory_path, model_weights_path, output_directory, min_detections, confidence_threshold)
