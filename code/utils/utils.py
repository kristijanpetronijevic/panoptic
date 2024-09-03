import json
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import albumentations as A

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    A.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def plot_object_distribution(annotation_file, title="Number of Objects per Class"):
    """
    Crta dijagram broja objekata po klasama za dati anotacioni fajl u COCO formatu.

    :param annotation_file: Putanja do JSON fajla sa anotacijama.
    :param title: Naslov dijagrama. Default vrednost je "Number of Objects per Class".
    """

    with open(annotation_file) as f:
        data = json.load(f)

    class_counts = defaultdict(int)
    for annotation in data['annotations']:
        class_id = annotation['category_id']
        class_counts[class_id] += 1

    class_names = {category['id']: category['name'] for category in data['categories']}
    class_counts_named = {class_names[k]: v for k, v in class_counts.items()}
    sorted_classes = sorted(class_counts_named.items(), key=lambda x: x[1], reverse=True)

    classes, counts = zip(*sorted_classes)
    plt.bar(classes, counts)
    plt.xticks(rotation=90)
    plt.xlabel('Classes')
    plt.ylabel('Number of Objects')
    plt.title(title)
    plt.show()
    
def load_coco_annotations(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_coco_annotations(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def merge_coco_annotations(file1, file2):
    data1 = load_coco_annotations(file1)
    data2 = load_coco_annotations(file2)

    # Combine images and annotations
    merged_data = {
        "images": data1["images"] + data2["images"],
        "annotations": data1["annotations"] + data2["annotations"],
        "categories": data1["categories"]
    }

    # Update annotation IDs and image IDs to be unique
    max_image_id = max(img["id"] for img in data1["images"])
    max_annotation_id = max(ann["id"] for ann in data1["annotations"])

    for ann in merged_data["annotations"][len(data1["annotations"]):]:
        max_annotation_id += 1
        ann["id"] = max_annotation_id

    return merged_data

def split_coco_dataset(coco_data, train_ratio=0.9):
    images = coco_data["images"]
    train_images, val_images = train_test_split(images, train_size=train_ratio)

    train_image_ids = {img["id"] for img in train_images}
    val_image_ids = {img["id"] for img in val_images}

    train_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in train_image_ids]
    val_annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] in val_image_ids]

    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data["categories"]
    }

    val_data = {
        "images": val_images,
        "annotations": val_annotations,
        "categories": coco_data["categories"]
    }

    return train_data, val_data

def save_model(model, optimizer, val_losses, train_losses, lr, epoch, batch_size, model_architecture, path, train_dataset_size, val_dataset_size, map_values_bbox,
                 model_name, seed=None, optimizer_hyperparameters=None):
                    
                    
    save_metadata(map_values_bbox, val_losses, train_losses, lr, epoch, batch_size, model_architecture, path,
                    train_dataset_size, val_dataset_size, model_name, seed, optimizer_hyperparameters)
    
    save_model_weights(model, optimizer, path, model_name)
    

def save_metadata(map_values_bbox, map_values_mask, val_losses, train_losses, lr, epoch, batch_size, model_architecture, 
                    path, train_dataset_size, val_dataset_size, model_name, seed=None, optimizer_hyperparameters=None):
                        
    # Pripremamo dictionary sa svim metapodacima
    metadata = {
        'val_losses': val_losses,
        'train_losses': train_losses,
        'mAP_values_bbox': map_values_bbox,
        'learning_rate': lr,
        'epoch': epoch,
        'batch_size': batch_size,
        'model_architecture': model_architecture,
        'train_dataset_size': train_dataset_size,
        'val_dataset_size': val_dataset_size,
        'random_seed': seed,
        'optimizer_hyperparameters': optimizer_hyperparameters
    }
    
    # Sačuvaj metapodatke u JSON formatu
    with open(f"{path + model_name}.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metapodaci sačuvani u: {path}.json")
    
    
def save_model_weights(model, optimizer, path, model_name):
    model_weights = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(model_weights, f"{path}.pth")
    print(f"Model i optimizer sačuvani u: {path + model_name}.pth")
    
    