import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from models.classifier import custom_CNN
from matplotlib import pyplot as plt
from torchvision.models import vgg16


def load_vgg_model(num_classes):
    model = vgg16(weights=True)

    for param in model.features.parameters():
        param.requires_grad = True

    model.classifier[6] = nn.Linear(4096, num_classes)
    return model

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_optimizer(name, model, training_config):
    if name == "adam":
        return optim.Adam(model.parameters(), lr=training_config.get("learning_rate"), weight_decay=training_config.get("weight_decay"))
    elif name == "sgd":
        return optim.SGD(model.parameters(), lr=training_config.get("learning_rate"),
                         momentum=training_config.get("momentum"), weight_decay=training_config.get("weight_decay"))
    else:
        raise ValueError("optimizer not implemented")

def train(config):
    if config.get("model").get("model_name") == "CustomCNN":
        model = custom_CNN()
    elif config.get("model").get("model_name") == "VGG16":
        model = load_vgg_model(num_classes=config.get("model").get("output_dim"))
    else:
        raise ValueError("model not implemented")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader, _ = load_svhn_data(batch_size=config.get("training").get("batch_size"))

    if config.get("loss").get("loss_type") == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("loss type not implemented")

    optimizer = get_optimizer(config.get("training").get("optimizer"), model, config.get("training"))

    best_val_loss = float("inf")
    patience = config.get("early_stopping").get("patience")
    wait = 0

    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(config.get("training").get("epochs")):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= len(val_loader)
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{config.get('training').get('epochs')}]: "
              f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}")

        if config.get("early_stopping").get("active"):
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
                if config["saving"]["enabled"]:
                    os.makedirs(config["saving"]["dir"], exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(config.get("saving").get("dir"), config.get("saving").get("name")))
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping.")
                    break

    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("saved_models/" + config.get("saving").get("loss_name"))

    plt.figure()
    plt.plot(train_accuracies, label="Train Acc")
    plt.plot(val_accuracies, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.savefig("saved_models/" + config.get("saving").get("accuracy_name"))

def load_svhn_data(data_dir="./data", batch_size=128, val_split=0.2):
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    base_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ])

    full_train = datasets.SVHN(root=data_dir, split='train', download=True, transform=train_transform)
    test = datasets.SVHN(root=data_dir, split='test', download=True, transform=base_transform)

    val_size = int(val_split * len(full_train))
    train_size = len(full_train) - val_size

    train, val = random_split(full_train, [train_size, val_size],
                              generator=torch.Generator().manual_seed(42))

    val.dataset.transform = base_transform

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    config = load_config("config/config_CNN.yml")
    train(config)