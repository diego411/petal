import torchvision.models as models
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.distance import cosine, euclidean
from itertools import combinations
import os
import numpy as np


def get_intra_class_similarity(class_label: str):
    features_list = []
    folder = f'../data/spectrograms/{class_label}'
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        feature_vector = extract_features(img_path)
        features_list.append(feature_vector)

    features_array = np.array(features_list)

    num_images = len(features_list)
    similarity_matrix = np.zeros((num_images, num_images))

    for i in range(num_images):
        for j in range(num_images):
            similarity_matrix[i, j] = 1 - cosine(features_array[i], features_array[j])  # Cosine similarity

    # Print average similarity in the class
    avg_similarity = np.mean(similarity_matrix)
    print(f"Average Cosine Similarity for class {class_label}: {avg_similarity:.4f}")


def get_inter_class_similarity():
    class_folders = [
        "../data/spectrograms/angry",
        "../data/spectrograms/disgusted",
        "../data/spectrograms/fearful",
        "../data/spectrograms/happy",
        "../data/spectrograms/neutral",
        "../data/spectrograms/sad",
        "../data/spectrograms/surprised",
    ]
    class_features = {}

    for class_folder in class_folders:
        class_name = os.path.basename(class_folder)
        print(f"Computing features for class: {class_name}")
        features_list = []

        for img_name in os.listdir(class_folder):
            img_path = os.path.join(class_folder, img_name)
            feature_vector = extract_features(img_path)
            features_list.append(feature_vector)

        class_features[class_name] = np.array(features_list)  # Store feature vectors for each class

    inter_class_similarity = {}

    for class_a, class_b in combinations(class_features.keys(), 2):
        print(f"Computing inter class similarity between: {class_a} and {class_b}")
        features_a = class_features[class_a]
        features_b = class_features[class_b]
        similarities = []

        for i in range(features_a.shape[0]):
            for j in range(features_b.shape[0]):
                sim = 1 - cosine(features_a[i], features_b[j])
                similarities.append(sim)

        inter_class_similarity[f"{class_a} vs {class_b}"] = np.mean(similarities)

    print("\nInter-Class Similarity:")
    for pair, sim in inter_class_similarity.items():
        print(f"{pair}: {sim:.4f}")


def get_resnet():
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
    model.eval()  # Set model to evaluation mode
    return model


def extract_features(image_path, model=get_resnet()):
    image = Image.open(image_path).convert('RGB')  # Convert to RGB if grayscale
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)  # Extract features
    return features.squeeze().numpy().flatten()
