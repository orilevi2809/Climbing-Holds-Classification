from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torchvision import models, transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import ToPILImage
import time
from torch.optim import lr_scheduler
import os
import cv2
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
class ImagePathDataset(Dataset):
    def __init__(self, images, labels, paths):
        self.images = images
        self.labels = labels
        self.paths = paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx], self.paths[idx]


misclassified_dir = "misclassified_images"
if not os.path.exists(misclassified_dir):
    os.makedirs(misclassified_dir)
IMG_SIZE = 224 #TODO: check if we can minize the size of the image e.g. 32*32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("cuda" if torch.cuda.is_available() else "cpu")
#img_dir = "../../labels/type_labels_base/"
img_dir = "../../../my_wall/labels/type_labels_base"
labels_dict = {"crimp": 0, "pinch": 1, "pocket": 2, "sloper": 3, "jug": 4}#,"jib":5}#, "edge": 5 , : 2
inv_labels_dict = {0 : "crimp",  1 : "pinch",  2 : "pocket",  3: "sloper",   4 : "jug"}#,5:"jib"}#, "edge": 5 ,

#labels_dict = {"low": 0, "medium": 1, "high": 2}
# Define the transform for image preprocessing
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=1),
    transforms.RandomRotation(degrees=(0, 30)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
noise_factor = 0.3  # Define the intensity of the noise

train_transform2 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=.3, hue=.3),
    transforms.RandomRotation(degrees=(-40, 40)),
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor: tensor + torch.randn_like(tensor) * noise_factor),  # Modified lambda function to add noise to a Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_transform4 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=.3, hue=.4),
    transforms.RandomAutocontrast(),
    transforms.RandomAdjustSharpness(sharpness_factor=2),
    transforms.RandomRotation(degrees=(-50, 50)),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_transform5 = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ColorJitter(brightness=.3, hue=.3),
    transforms.RandomRotation(degrees=(-25, 25)),
    transforms.ToTensor(),
    # Modified lambda function to add noise to a Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def save_model(model, model_file):
    # Save the model's state dictionary to disk
    torch.save(model.state_dict(), model_file)

def load_images_paths(img_dir, labels):
    data = []
    target = []
    label_count = {}  # Dictionary to store the count of images for each label
    for label in labels:
        label_dir = os.path.join(img_dir, str(label))
        count = 0  # Initialize the count for the current label
        for filename in os.listdir(label_dir):
            if filename.endswith(".jpg") or filename.endswith(".jpeg")or filename.endswith(".png"):
                img_path = os.path.join(label_dir, filename)
                data.append(img_path)
                target.append(labels_dict[label])
                count += 1  # Increment the count for the current label
        label_count[label] = count  # Store the count in the dictionary

    # Print the count of images for each label
    for label, count in label_count.items():
        print(f"Number of images for label '{label}': {count}")
    return data, target

def get_model_densenet():
    # Load pre-trained DenseNet model and replace the classifier layer
    model = models.densenet201(pretrained=True)


    dropout_rate = 0.3  # Specify the dropout rate
    additional_fc_layer = nn.Sequential(
      nn.Dropout(dropout_rate),
      nn.Linear(model.classifier.in_features, len(labels_dict)),  # Replace len(labels_dict) with the number of output classes
      nn.Softmax(dim=1)  # Add the softmax activation layer
    )
    model.classifier = additional_fc_layer  # assuming labels_dict is defined elsewhere

    model = model.to(device)
    return model

class CustomLayer(nn.Module):
    def __init__(self, dropout_rate, in_features, out_features):
        super().__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(in_features, out_features) # Change this
        self.softmax = nn.Softmax(dim=1)
        self.out_features = out_features

    def forward(self, x):
        x = self.dropout(x)
        x = self.linear(x) # Change this
        #x = self.softmax(x)
        return x
def get_model_mobileNet():
    # Load pre-trained MobileNetV2 model and replace the classifier layer
    model = models.mobilenet_v2(pretrained=True)
    dropout_rate = 0.3  # Specify the dropout rate
    in_features = model.classifier[-1].in_features  # Replace with the correct number of input features
    out_features = len(labels_dict)  # Replace with the correct number of output features
    additional_fc_layer = CustomLayer(dropout_rate, in_features, out_features)

    model.classifier[-1] = additional_fc_layer

    model = model.to(device)
    return model

def load_image_data(img_paths, labels, transform, augment_unbalanced=False):
    if augment_unbalanced:
        data = []
        target = []
        path_list = []
        label_counts = {label: 0 for label in set(labels)}
        for img_path, label in zip(img_paths, labels):
            img = Image.open(img_path).convert("RGB")
            img = test_transform(img)
            data.append(img)
            path_list.append(img)
            target.append(label)
            label_counts[label] += 1

        is_unbalanced = any(count < max(label_counts.values()) for count in label_counts.values())
        print(label_counts)

        if is_unbalanced:
            print("unbalanced")
            augmented_data = []
            augmented_target = []
            path_list = []
            #TODO: add while all equal
            max_count = max(label_counts.values())
            for img_path, label in zip(img_paths, labels):
                img = Image.open(img_path).convert("RGB")
                augmented_data.append(test_transform(img))
                augmented_target.append(label)
                path_list.append(img_path)
                for i in range(1):
                    if label_counts[label] < 1*3*max_count:
                        augmented_img = train_transform4(img)
                        augmented_data.append(augmented_img)
                        augmented_target.append(label)
                        path_list.append(img_path)
                        label_counts[label] += 1
                    if label_counts[label] < 1*3*max_count:
                        augmented_img = train_transform(img)
                        augmented_data.append(augmented_img)
                        augmented_target.append(label)
                        path_list.append(img_path)
                        label_counts[label] += 1
                    if label_counts[label] < 1*3*max_count:
                        augmented_img = train_transform2(img)
                        augmented_data.append(augmented_img)
                        augmented_target.append(label)
                        path_list.append(img_path)
                        label_counts[label] += 1
                    if label_counts[label] < 1*3*max_count:
                        augmented_img = train_transform5(img)
                        augmented_data.append(augmented_img)
                        augmented_target.append(label)
                        path_list.append(img_path)
                        label_counts[label] += 1
            print(label_counts)
            return augmented_data, augmented_target,path_list
        return data, target,path_list
    else:
        if isinstance(img_paths[0], torch.Tensor):  # Check if data is already in tensor format
            return img_paths, labels, img_paths
        else:
            data = []
            target = []
            path_list = []
            for img_path, label in zip(img_paths, labels):
                img = Image.open(img_path).convert("RGB")
                img = test_transform(img)
                data.append(img)
                target.append(label)
                path_list.append(img_path)
            return data, target, path_list

def train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device,scheduler,epochs):
    train_loss = []
    vals_loss = []
    for epoch in range(epochs):  # 10 is the best!!
        running_loss = 0.0
        total_correct = 0
        total_samples = 0

        # Switch model to training mode
        model.train()

        for i, data in enumerate(train_dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            # Calculate the number of correct predictions in the current batch
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        print(f"Epoch: {epoch + 1}, Learning Rate: {scheduler.get_last_lr()}")
        scheduler.step()
        train_accuracy = 100 * total_correct / total_samples

        # Switch model to evaluation mode for validation
        model.eval()

        val_correct = 0
        val_samples = 0
        val_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(val_dataloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels).sum().item()
                val_samples += labels.size(0)

        val_accuracy = 100 * val_correct / val_samples

        print(f'Epoch: {epoch + 1}, Training loss: {running_loss / len(train_dataloader)}, Training accuracy: {train_accuracy}%, Validation loss: {val_loss / len(val_dataloader)}, Validation accuracy: {val_accuracy}%')
        train_loss.append(running_loss / len(train_dataloader))
        vals_loss.append(val_loss / len(val_dataloader))
    print('Finished Training')
    epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plt.plot(epochs, vals_loss, 'r', label='Validation Loss')
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    return model


def plot_image(image,path, label, prediction, inverse_normalize):
    to_pil_transform = ToPILImage()

    image = inverse_normalize(image)  # Invert the normalization
    image = to_pil_transform(image)  # Convert tensor image to PIL image

    image = np.array(image)  # Convert PIL image to numpy array
    plt.imshow(image)
    plt.axis('off')
    plt.text(10, 10, f"Label: {label}", color='red', fontsize=12)
    plt.text(10, 30, f"Prediction: {prediction}", color='red', fontsize=12)
    plt.text(10, 50, f"{path}", color='red', fontsize=12)
    plt.show()


def evaluate_classifier(test_dataloader, model, device,img_paths_test):
    inverse_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                              std=[1 / 0.229, 1 / 0.224, 1 / 0.225])

    model.eval()  # Switch model to evaluation mode
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels, paths in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if not torch.all(predicted == labels):
                filtered_imgs = images[predicted != labels]
                incorrect_indices = (predicted != labels).nonzero().squeeze()
                if incorrect_indices.dim() == 0:
                    incorrect_indices = incorrect_indices.unsqueeze(
                        0)  # Convert the 0D tensor to a 1D tensor with one element
                mispredicted_paths = [paths[i] for i in incorrect_indices]
                for i in range(len(filtered_imgs)):
                    mispredicted_image = filtered_imgs[i]  # Select the first mispredicted image
                    mispredicted_path = mispredicted_paths[i]
                    label = labels[predicted != labels][i].item()
                    prediction = predicted[predicted != labels][i].item()
                    path = mispredicted_path.split('/')[-1]
                    plot_image(mispredicted_image,path, label, prediction, inverse_normalize)

            # Move tensors to CPU before converting to numpy arrays
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    # Generate classification report and confusion matrix
    report = classification_report(all_labels, all_predictions, zero_division=1)
    matrix = confusion_matrix(all_labels, all_predictions)
    print("Classification Report:")
    print(report)
    print("Confusion Matrix:")
    print(matrix)




def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = test_transform(img_pil)
        preprocessed_images.append(img_tensor)
    return torch.stack(preprocessed_images)

def predict_labels(images, model):
    with torch.no_grad():
        transformed_images = preprocess_images(images)
        outputs = model(transformed_images)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probabilities, dim=1)
        most_probable_label = list(labels_dict.keys())[predicted.item()]
        return most_probable_label

def preprocess_images(images):
    preprocessed_images = []
    for img in images:
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        img_tensor = test_transform(img_pil)
        preprocessed_images.append(img_tensor)
    preprocessed_images = torch.stack(preprocessed_images)
    # Normalize the images using the same statistics as BatchNorm
    #preprocessed_images = nn.functional.normalize(preprocessed_images, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return preprocessed_images


def get_module_path(model, module):
    for name, child in model.named_modules():
        if child is module:
            return name


def find_image_path(image1, label, prediction):
    # Initialize variables to keep track of the best match
    best_similarity = -1
    best_image_path = None
    folder1 = os.path.join(img_dir, inv_labels_dict[label])
    # Iterate over files in the folder1
    for file_name in os.listdir(folder1):
        # Construct the file path
        file_path = os.path.join(folder1, file_name)

        # Load the current image using cv2.imread()
        current_image = cv2.imread(file_path)

        # Calculate the SSIM similarity between image1 and current_image
        similarity = ssim(image1, current_image, multichannel=True)

        # Update the best match if the current similarity is higher
        if similarity > best_similarity:
            best_similarity = similarity
            best_image_path = file_path

    if best_image_path:
        # Copy the best matching image to the destination folder
        print("best_image_path= ",best_image_path )
        print("label= ", prediction)
        print("label= ", prediction)
    else:
        print("No similar image found in folder1.")
        return None






def save_model_if_true(model,true = False):
    if true:
        # Save the MobileNetV2 model
        model_file = 'hold_classifier_MobileNet.pth'
        save_model(model, model_file)

        # Load the model for inference
        model.load_state_dict(torch.load('hold_classifier_MobileNet.pth'))


def print_training_time(start_time):
    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    # save_mode(model)
    # Convert elapsed time to minutes
    elapsed_time_minutes = elapsed_time / 60
    # Print the elapsed time
    print("Time taken by train_classifier: {:.2f} minutes".format(elapsed_time_minutes))


def get_model_hyper(learning_rate, model):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    return criterion, optimizer, scheduler


def preprocces_data():
    ####### pre prosses data #TODO:refactor to preprocces
    img_paths, labels = load_images_paths(img_dir, labels_dict.keys())
    # Split the data into train, validation, and test sets (e.g., 70%, 15%, 15%)
    img_paths_train, img_paths_test, labels_train, labels_test = train_test_split(img_paths, labels, test_size=0.3,
                                                                                  stratify=labels, random_state=42)
    img_paths_val, img_paths_test, labels_val, labels_test = train_test_split(img_paths_test, labels_test,
                                                                              test_size=0.5, stratify=labels_test,
                                                                              random_state=42)
    data_train, labels_train, img_paths_train = load_image_data(img_paths_train, labels_train, train_transform,
                                                                augment_unbalanced=True)
    data_val, labels_val, img_paths_val = load_image_data(img_paths_val, labels_val, test_transform,
                                                          augment_unbalanced=False)
    data_test1, labels_test1, img_paths_test = load_image_data(img_paths_test, labels_test, test_transform,
                                                               augment_unbalanced=False)
    # Convert to tensors
    data_train = torch.stack(data_train)
    labels_train = torch.tensor(labels_train)
    data_val = torch.stack(data_val)
    labels_val = torch.tensor(labels_val)
    data_test = torch.stack(data_test1)
    labels_test = torch.tensor(labels_test1)
    # Replace TensorDataset with ImagePathDataset
    train_dataset = ImagePathDataset(data_train, labels_train, img_paths_train)
    val_dataset = ImagePathDataset(data_val, labels_val, img_paths_val)
    test_dataset = ImagePathDataset(data_test, labels_test, img_paths_test)
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return img_paths_test, test_dataloader, train_dataloader, val_dataloader


def predict_unlabeled(model, device):
    src_imgs_path = "../../labels/type_labels_base/"
    print()
    model.eval()  # Switch model to evaluation mode
    inverse_normalize = transforms.Normalize(mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
                                             std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
    to_pil_transform = transforms.ToPILImage()

    # Iterate through the directory
    for dir in os.listdir(src_imgs_path):
        if (dir.startswith("unknown_")):
            dir_path = os.path.join(src_imgs_path, dir)
            for img_name in os.listdir(dir_path):
                if (img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png")) :
                    img_path = os.path.join(dir_path, img_name)
                    # Load image
                    image = Image.open(img_path).convert("RGB")
                    # Apply the same transformations that you used for test data
                    image_tensor = test_transform(image).unsqueeze(0).to(device)  # Add batch dimension

                    # Predict
                    with torch.no_grad():
                        outputs = model(image_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(probabilities, dim=1)
                        predicted_label = inv_labels_dict[predicted.item()]

                    # Plot
                    image_tensor = image_tensor.squeeze(0)  # Remove batch dimension
                    image_tensor = inverse_normalize(image_tensor).cpu()  # Invert the normalization

                    image = to_pil_transform(image_tensor)  # Convert tensor image to PIL image

                    plt.imshow(image)
                    plt.axis('off')
                    plt.text(10, 10, f"Prediction: {predicted_label}", color='red', fontsize=12)
                    plt.text(10, 30, f"dir: {dir}", color='red', fontsize=12)
                    plt.text(10, 50, f"img_name: {img_name}", color='red', fontsize=12)

                    plt.show()


def main():
    img_paths_test, test_dataloader, train_dataloader, val_dataloader = preprocces_data()

    # Load pre-trained MobileNetV2 model and replace the classifier layer
    model = get_model_mobileNet()

    # Define loss criterion and optimizer
    learning_rate = 0.00005
    criterion, optimizer, scheduler = get_model_hyper(learning_rate, model)

    # Get the current time
    start_time = time.time()
    # Call the train_classifier function
    epochs = 10
    model = train_classifier(train_dataloader, val_dataloader, model, criterion, optimizer, device, scheduler, epochs)

    print_training_time(start_time)

    save_model_if_true(model,False)

    evaluate_classifier(test_dataloader, model, device,img_paths_test)
    print(labels_dict)
    #predict_unlabeled(model, device)

if __name__ == '__main__':
    main()
