import os
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
from torchvision import models, transforms
from PIL import Image

import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="Path to the input image")
    return parser.parse_args()

class HoldNode:
    def __init__(self, hold_type, center ,pred_index,size,percentage):
        self.type = hold_type
        self.center = center
        self.pred_index = pred_index
        self.size = size
        self.percentage = percentage


    def __str__(self):
        return f"Type: {self.type} \nCenter: {self.center}\nIndex: {self.pred_index}\n"

labels_dict = {0: "crimp", 1: "pinch", 2: "pocket", 3: "sloper", 4: "jug"}
def config_model():
    # Get config and weigths for model
    cfg = get_cfg()
    cfg.merge_from_file("../../yml/experiment_config.yml")
    cfg.MODEL.WEIGHTS = "../../ml/weights/model_final.pth"
    cfg.MODEL.DEVICE = 'cpu'
    # Set metadata, in this case only the class names for plotting
    MetadataCatalog.get("meta").thing_classes = ["hold", "volume"]
    metadata = MetadataCatalog.get("meta")

    predictor = DefaultPredictor(cfg)
    return predictor, metadata
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_image_instances(predictor,img_path):
    img = cv2.imread(img_path)
    # Use predictor to detect objects in image
    # best_params = find_best_parameter(img,predictor,metadata)
    # best_img = get_best_image(img,best_params)
    outputs = predictor(img)
    # Get the detected instances
    instances = outputs["instances"]
    predictors = instances.to("cpu")
    return img, instances, predictors


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    ])
    return preprocess(image)


def predict_image(model, input_batch, labels_dicts):
    if input_batch is None:
        return None, None

    with torch.no_grad():
        # Preprocess the images if necessary
        if isinstance(input_batch[0], Image.Image):
            input_batch = torch.stack([preprocess_image(image) for image in input_batch])
        else:
            input_batch = torch.stack(input_batch)

        # Get the raw output from the model
        output = model(input_batch)

        # Convert raw output to probabilities using softmax
        probabilities = F.softmax(output, dim=1)

        # Get the index of the max probability
        _, predicted_indices = torch.max(output, 1)
        # Get the predicted label
        predicted_label = labels_dicts[predicted_indices[0].item()]

        # Get the confidence score for the predicted label
        confidence_score = probabilities[0][predicted_indices[0]].item()

    return predicted_label, int(confidence_score*100)


def create_poly_mask(img, instances, dilation_size=10):
    # Create a mask for the polygons
    mask = np.zeros_like(img)

    # Draw the polygons on the mask
    pred_masks = instances.pred_masks
    for i in range(pred_masks.shape[0]):
        # Get the predicted polygon
        pred_polygon = pred_masks[i].numpy().astype(np.uint8) * 255

        # Reshape the pred_polygon to have three channels
        pred_polygon = np.stack([pred_polygon] * 3, axis=-1)

        # Merge the polygon with the mask
        mask = cv2.bitwise_or(mask, pred_polygon)

    # Dilate the mask to expand the polygon boundaries
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    mask = cv2.dilate(mask, kernel)

    return mask
def filter_instances_by_minimum_size(instances, pred_boxes):
    # Create a boolean mask to filter out the predictions whose sizes are greater than 200
    widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    size_mask = widths * heights
    instances = instances[200 < size_mask]
    return instances
def get_contour_poly_mask(poly_mask):
    # Convert the polygon mask to grayscale
    poly_gray = cv2.cvtColor(poly_mask, cv2.COLOR_BGR2GRAY)
    # plot_image(poly_gray)
    # Find the contours of the polygon
    contours, hierarchy = cv2.findContours(poly_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours

def draw_polygon(image, poly_mask, color=(255, 255, 255)):
    if np.sum(poly_mask) == 0:
        return image

    # Convert the polygon mask to grayscale
    poly_gray = cv2.cvtColor(poly_mask, cv2.COLOR_BGR2GRAY)

    # Find the contours of the polygon
    contours, hierarchy = cv2.findContours(poly_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Create a copy of the image to draw the contours on
    image_with_contours = image.copy()

    # Draw the contours on the image
    cv2.drawContours(image_with_contours, contours, -1, color, thickness=3)

    return image_with_contours

def draw_and_save_image(metadata, predictions,holds, img, mask,start_hold_mask,end_hold_mask,legs_hold_mask,draw_bw = False,show_label_mode = True):
    instances = predictions.to("cpu")

    # Get the predicted bounding boxes
    pred_boxes = instances.pred_boxes.tensor.cpu().numpy()

    instances = filter_instances_by_minimum_size(instances, pred_boxes)
    # Loop over the predicted instances and classify each object
    sizes = []
    text_list = []
    position_list = []
    mean_size = 0
    ins_num = 0
    for i in range(len(instances)):
        instance = instances[i]
        bbox = instance.pred_boxes.tensor.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        cur_mask = instance.pred_masks.cpu().numpy().astype(np.uint8)[0]
        cur_mask = cur_mask[y1:y2, x1:x2]
        cur_mask = np.where(cur_mask > 0, 255, 0).astype(np.uint8)
        # Calculate the area inside the mask
        area = np.sum(cur_mask == 255)
        # Apply Canny edge detection
        sizes.append(area)
        ins_num += 1
    median_size = np.median(sizes)

    for i in range(len(instances)):
        instance = instances[i]
        bbox = instance.pred_boxes.tensor.cpu().numpy()[0]
        x1, y1, x2, y2 = map(int, bbox)


        v = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE_BW)
        x1, y1, x2, y2 = map(int, bbox)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        size = np.abs((x1 - x2)) * np.abs(y1 - y2)
        object_img = img[y1:y2, x1:x2]
        object_img = cv2.resize(object_img, (128, 128))
        # Predict the label for the object using the SVM model
        label = holds[i].type
        percentage = holds[i].percentage
        #volume_type = holds[i].volume_type

        #if median_size > size:
        #    if label == "sloper":
        #        label = "pinch"
        #else:
        #    if label == "jug":
        #        label = "sloper"
        #    elif label == "crimp":
        #        label = "pinch"
        #    elif label == "pocket":
        #        label = "sloper"
        #    elif label == "pinch":
        #        label = "sloper"

        # Draw the instance edges on the image using the predicted label and probability



        text_list.append(f"{label} {percentage}%")
        position_list.append((x1, y1 - 20))



    modified_img = v.output.get_image()[:, :, ::-1]




    modified_img = draw_polygon(modified_img, mask,(255, 255, 255))
    modified_img = draw_polygon(modified_img, start_hold_mask,(0, 0, 255)) #red
    modified_img = draw_polygon(modified_img, end_hold_mask,(0, 0, 255)) #red
    modified_img = draw_polygon(modified_img, legs_hold_mask,(0, 255, 255))#yellow
    if (show_label_mode):
        v = Visualizer(modified_img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE_BW)

        for i in range(len(text_list)):
            v.draw_text(text_list[i], position_list[i], font_size=15, color="r")
        modified_img = v.output.get_image()[:, :, ::-1]
        #modified_img = cv2.cvtColor(modified_img, cv2.COLOR_RGB2BGR)
    return modified_img

def get_holds(img,predictions,instances,model):
    print(len(instances))
    holds = []
    sizes = []
    mean_hold_size = 0
    bbox_cnt = 0
    volumes = []
    for i in range(len(instances)):
        box = predictions.pred_boxes[i].tensor.cpu().numpy()
        x1, y1, x2, y2 = map(int, box[0])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.shape[1], x2)
        y2 = min(img.shape[0], y2)
        object_img = img[y1:y2, x1:x2]

        cur_mask = instances[i].pred_masks.cpu().numpy().astype(np.uint8)[0]
        cur_mask = cur_mask[y1:y2, x1:x2]
        cur_mask = np.where(cur_mask > 0, 255, 0).astype(np.uint8)

        # Create a copy of the original image
        filtered_img = object_img.copy()



        # Erase what is outside of the edges in the filtered image
        filtered_img[cur_mask == 0] = 0




        size = np.abs((x1 - x2)) * np.abs(y1 - y2)
        sizes.append(size)
        mean_hold_size += size
        bbox_cnt += 1
    mean_hold_size = mean_hold_size/bbox_cnt
    median_size = np.median(sizes)
    min_size = min(sizes)
    max_size = max(sizes)
    print("min(sizes)= ",min(sizes))
    print("max(sizes)= ",max(sizes))
    print("mean_hold_size= ",mean_hold_size)
    print("median_size= ", median_size)
    modified_instances = []
    print(len(predictions))
    for i in range(len(predictions)):
        box = predictions.pred_boxes[i].tensor.cpu().numpy()
        x1, y1, x2, y2 = map(int, box[0])
        x1 = max(0, x1 )
        y1 = max(0, y1 )
        x2 = min(img.shape[1], x2 )
        y2 = min(img.shape[0], y2 )
        size = np.abs((x1-x2))*np.abs(y1-y2)

        object_img = img[y1:y2, x1:x2]
        object_img_pil = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))  # Convert NumPy array to PIL Image
        hold_type, confidence_score = predict_image(model, [object_img_pil],labels_dict)
        #volume_type = predict_image(model_volume,[object_img_pil],labels_dict_volume)
        #print(i)
        #plot_image(object_img,hold_type,confidence_score)
        #if median_size*4 > size:
        #    if hold_type == "sloper":
        #       hold_type = "pinch"
        #    elif hold_type == "edge":
        #        hold_type = "pinch"

            #if hold_type == "crimp" and volume_type == "high":
            #    hold_type == "jug"
        #    if hold_type == "jug" and volume_type == "low":
        #        hold_type = "pinch"
        #else:
        #    if hold_type == "crimp":
        #        hold_type = "edge"


        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        #angle_degrees = predict_image(model_angle,[object_img_pil],labels_dict_angle)

        hold = HoldNode(hold_type, center,i,size, confidence_score)
        holds.append(hold)
    print(len(holds))




    return holds

def plot_image(img, str, num):
    img_mat = cv2.UMat(img)
    combined_string = "{}{}".format(str, num)    # Add the text string to the image
    cv2.putText(img_mat, combined_string, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # Display the image
    plt.imshow(cv2.cvtColor(img_mat.get(), cv2.COLOR_BGR2RGB))
    plt.show()


def get_euclidean_distance(box1, box2):
    """Calculates the Euclidean distance between the centers of two boxes"""
    center1 = np.mean(box1, axis=0)
    center2 = np.mean(box2, axis=0)
    return np.linalg.norm(center1 - center2)


def plot_route(g, route):
    print("route = ", route)
    edges = [(route[i], route[i+1]) for i in range(len(route) - 1)]
    # Draw the edges of the graph with arrows
    nx.draw_networkx_edges(g, pos=nx.spring_layout(g), edgelist=edges, arrows=True)
    plt.show()


def load_mobilenet_model(num_classes):
    model = models.mobilenet_v2(pretrained=False)  # Instantiate the model architecture

    dropout_rate = 0.3  # Specify the dropout rate
    model.classifier[-1] = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.classifier[-1].in_features, len(labels_dict)),  # Replace len(labels_dict) with the number of output classes
        nn.Softmax(dim=1)  # Add the softmax activation layer
    )
    checkpoint = torch.load('../classification_hold_type/hold_classifier_MobileNet.pth')

    # Get the model state_dict from the checkpoint
    if 'model_state_dict' in checkpoint:
        checkpoint = checkpoint['model_state_dict']

    # Filter out the keys that don't match the model architecture
    checkpoint = {k: v for k, v in checkpoint.items() if k in model.state_dict()}

    # Load the filtered state_dict into the model
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model




def main():
    args = parse_arguments()
    print(args.image_path)

    image_path = os.path.abspath(args.image_path)  # Make image_path absolute

    predictor, metadata = config_model()
    classification_model = load_mobilenet_model(len(labels_dict))
    #image_path = '../../images/original_spraywall_img_corrected.png'
    img, instances, predictors = get_image_instances(predictor, image_path)
    plot_image(img, "graph original image:  ", 1)

    holds = get_holds(img,predictors,instances,classification_model)
    print(len(holds))
    route_mask = create_poly_mask(img, instances)
    route_image = draw_and_save_image(metadata, predictors, holds, img, route_mask, [],[], [],False,False)
    #route_image = cv2.cvtColor(route_image, cv2.COLOR_RGB2BGR)
    plot_image(route_image, "graph num:  ", 1)
    route_image = draw_and_save_image(metadata, predictors, holds, img, route_mask, [],[], [],False,True)
    #route_image = cv2.cvtColor(route_image, cv2.COLOR_RGB2BGR)
    plot_image(route_image, "graph num:  ", 1)
    print(route_image.size)
    cv2.imwrite(f"../../images/output_image_1.png",route_image)


if __name__ == '__main__':
    main()