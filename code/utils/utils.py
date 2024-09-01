import json
import matplotlib.pyplot as plt
from collections import defaultdict

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
    

def save_model(model, optimizer, val_losses, train_losses, lr, epoch, batch_size, model_architecture, path, train_dataset_size, val_dataset_size, map_values,
                model_name, seed=None, optimizer_hyperparameters=None):
                    
    save_metadata(map_values,val_losses, train_losses, lr, epoch, batch_size, model_architecture, path,
                    train_dataset_size, val_dataset_size, model_name, seed, optimizer_hyperparameters)
    
    save_model_weights(model, optimizer, path, model_name)
    

def save_metadata(map_values,val_losses, train_losses, lr, epoch, batch_size, model_architecture, 
                    path, train_dataset_size, val_dataset_size, model_name, seed=None, optimizer_hyperparameters=None):
                        
    # Pripremamo dictionary sa svim metapodacima
    metadata = {
        'val_losses': val_losses,
        'train_losses': train_losses,
        'mAP_values': map_values,
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
    
    