import json
import matplotlib.pyplot as plt
from collections import defaultdict
import random
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import albumentations as A
import pandas as pd

def set_seed(seed):
    
    """
    Postavlja zadati seed za replikaciju eksperimenta.
    :param seed: Vrednost random seed-a koja će biti postavljena
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
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
    """
    Spaja anotacije iz dva COCO dataset fajla u jedan.

    Argumenti:
    - file1: putanja do prvog COCO dataset fajla
    - file2: putanja do drugog COCO dataset fajla

    Povratna vrednost:
    - merged_data: spojen dataset koji sadrži slike, anotacije i kategorije iz oba dataset-a
    """
    
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
    """
    Deli COCO dataset na trening i validacioni skup na osnovu zadatog odnosa.

    Argumenti:
    - coco_data: COCO dataset koji sadrži ključeve "images", "annotations" i "categories"
    - train_ratio: udeo podataka koji će biti uključen u trening skup (podrazumevana vrednost je 0.9)

    Povratna vrednost:
    - train_data: trening skup koji sadrži slike, anotacije i kategorije
    - val_data: validacioni skup koji sadrži slike, anotacije i kategorije
    """
    
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
                 model_name, backbone, seed=None, optimizer_hyperparameters=None, mention = None):
    
    """
    Čuva model, optimizator i metapodatke u odvojene fajlove.
    Funkcija prvo čuva metapodatke modela u JSON fajl koristeći `save_metadata` funkciju,
    a zatim čuva težine modela i stanje optimizatora koristeći `save_model_weights` funkciju.
    """
                    
    save_metadata(map_values_bbox, val_losses, train_losses, lr, epoch, batch_size, model_architecture, path,
                    train_dataset_size, val_dataset_size, model_name, backbone, seed, optimizer_hyperparameters, mention=mention)
    
    save_model_weights(model, optimizer, path, model_name)
    

def save_metadata(map_values_bbox, val_losses, train_losses, lr, epoch, batch_size, model_architecture, 
                    path, train_dataset_size, val_dataset_size, model_name, backbone, seed=None, optimizer_hyperparameters=None, mention = None):
    """
    Čuva metapodatke o treniranju modela u JSON formatu.

    Argumenti:
    - map_values_bbox: mAP vrednosti za bounding box
    - val_losses: gubici (loss) na validacionom skupu tokom epoha
    - train_losses: gubici (loss) na trening skupu tokom epoha
    - lr: vrednost stope učenja (learning rate)
    - epoch: trenutni broj epoha
    - batch_size: veličina batch-a korišćena tokom treniranja
    - model_architecture: arhitektura modela (npr. ime modela kao string)
    - path: putanja do direktorijuma gde će metapodaci biti sačuvani
    - train_dataset_size: veličina trening skupa
    - val_dataset_size: veličina validacionog skupa
    - model_name: naziv fajla za sačuvane metapodatke (bez ekstenzije)
    - backbone: ime korišćenog backbone-a (npr. ResNet, EfficientNet)
    - seed: opcioni random seed korišćen za replikaciju eksperimenta (podrazumevana vrednost je None)
    - optimizer_hyperparameters: opcioni hiperparametri korišćeni za optimizator (podrazumevana vrednost je None)
    - mention: opcionalna dodatna napomena ili komentar (podrazumevana vrednost je None)
    """
                  
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
        'optimizer_hyperparameters': optimizer_hyperparameters,
        'backbone': backbone,
        'mention':mention
    }
    
    # Sačuvaj metapodatke u JSON formatu
    with open(f"{path + model_name}.json", 'w') as json_file:
        json.dump(metadata, json_file, indent=4)
    print(f"Metapodaci sačuvani u: {path + model_name}.json")
    
    
def save_model_weights(model, optimizer, path, model_name):
    """
    Čuva težine modela i stanje optimizatora u fajl.

    Argumenti:
    - model: instanca modela čije će se stanje sačuvati
    - optimizer: instanca optimizatora čije će stanje biti sačuvano
    - path: putanja do direktorijuma gde će fajl biti sačuvan
    - model_name: naziv fajla u koji će se sačuvati kontrolna tačka (bez ekstenzije)
    """
    
    model_weights = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(model_weights, f"{path + model_name}.pth")
    print(f"Model i optimizer sačuvani u: {path + model_name}.pth")
    
def load_model_weights(model, path, model_name):
    """
    Učitava težine modela iz kontrolne tačke.

    Argumenti:
    - model: instanca modela čiji će se parametri učitati
    - path: putanja do direktorijuma gde se nalazi sačuvana kontrolna tačka
    - model_name: naziv fajla sa kontrolnom tačkom (bez ekstenzije)

    Povratna vrednost:
    - model: model sa učitanim parametrima
    """
    checkpoint = torch.load(f"{path + model_name}.pth")
    
    # Učitajte samo stanje modela
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model učitan iz: {path + model_name}.pth")
    return model


def load_model_weights_and_optimizer(model, optimizer, path, model_name):
    """
    Učitava težine modela i stanje optimizatora iz kontrolne tačke.

    Argumenti:
    - model: instanca modela čiji će se parametri učitati
    - optimizer: instanca optimizatora čije će stanje biti učitano
    - path: putanja do direktorijuma gde se nalazi sačuvana kontrolna tačka
    - model_name: naziv fajla sa kontrolnom tačkom (bez ekstenzije)

    Povratna vrednost:
    - model: model sa učitanim parametrima
    - optimizer: optimizator sa učitanim stanjem

    Ovaj metod učitava 'model_state_dict' i 'optimizer_state_dict' iz
    fajla sa ekstenzijom `.pth` i postavlja ih na dati model i optimizator.
    """
    
    checkpoint = torch.load(f"{path + model_name}.pth")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model i optimizer učitani iz: {path + model_name}.pth")
    return model, optimizer
    
def visualize_metrics(mAP, categories):
    """
    Vizualizuje rezultate iz MeanAveragePrecision.

    Parametri:
    - mAP: dict, rezultat funkcije MeanAveragePrecision.compute()
    - categories: dict, mapa ID-jeva klasa na imena klasa

    Povratna vrednost:
    - None
    """

    overall_metrics = {k: v for k, v in mAP.items() if not k.endswith('_per_class') and k != 'classes'}
    per_class_metrics = {k: v for k, v in mAP.items() if k.endswith('_per_class')}

    df_overall = pd.DataFrame(overall_metrics.items(), columns=['Metrička vrednost', 'Vrednost'])

    pd.set_option('display.float_format', lambda x: '%.4f' % x)

    print("Ukupne metrike:")
    print(df_overall.to_string(index=False))

    if per_class_metrics:
        num_classes = len(per_class_metrics['map_per_class'])
        class_ids = [idx + 1 for idx in range(num_classes)]  # Dodajemo 1 da bi se poklopilo sa ID-jevima klasa
        class_labels = [categories.get(class_id, f'Klasa {class_id}') for class_id in class_ids]

        df_per_class = pd.DataFrame({'ID klase': class_ids, 'Klasa': class_labels})
        for metric_name, values in per_class_metrics.items():
            metric_name = metric_name.replace('_per_class', '')
            df_per_class[metric_name] = values

        columns_order = ['ID klase', 'Klasa'] + sorted([col for col in df_per_class.columns if col not in ['ID klase', 'Klasa']])
        df_per_class = df_per_class[columns_order]

        print("\nMetrike po klasama:")
        print(df_per_class.to_string(index=False))

        plt.figure(figsize=(10, 6))
        plt.bar(df_per_class['Klasa'], df_per_class['map'])
        plt.xlabel('Klasa')
        plt.ylabel('mAP')
        plt.title('Mean Average Precision po klasama')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Nema dostupnih metrika po klasama.")

