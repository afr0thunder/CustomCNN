import os
import cv2
import yaml
import torch
import numpy as np
import torch.nn.functional as F
from data.preprocess import preprocess
from detect import detect_boxes
from models.classifier import custom_CNN
from torchvision.models import vgg16

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def draw_boxes(image, boxes, labels, confidences):
    temp_image = np.copy(image)
    valid = [(b, l, c) for b, l, c in zip(boxes, labels, confidences) if l != "?"]
    for (x, y, w, h), label, conf in valid:
        cv2.rectangle(temp_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = str(label)
        coor_x = x + 2
        coor_y = y + 15
        cv2.putText(temp_image, text, (coor_x, coor_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return temp_image

def load_model(name, path):
    if name == "customcnn":
        model = custom_CNN()
    elif name == "vgg16":
        model = vgg16(pretrained=False)
        model.classifier[6] = torch.nn.Linear(4096, 11)
    else:
        raise ValueError("model not implemented")

    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def classify_patch(image, boxes, eval_config, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    preprocess_config = load_config("config/config_pre.yml")

    digits = []
    confidences = []
    for (x, y, w, h) in boxes:
        patch = image[y:y + h, x:x + w]
        patch = preprocess(patch, preprocess_config)
        patch = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(patch)
            probs = F.softmax(output, dim=1)
            confidence, prediction = torch.max(probs, 1)

        if confidence.item() >= eval_config.get("inference").get("confidence_threshold"):
            digits.append(prediction.item())
            confidences.append(confidence.item())
        else:
            digits.append("?")
            confidences.append(confidence.item())

    return digits, confidences

def main():
    config = load_config("config/config_pre.yml")
    eval_config = load_config("config/config_evaluation.yml")
    model = load_model(eval_config.get("model"), eval_config.get("inference").get("model_path"))

    image_files = [f for f in sorted(os.listdir("dataset/train")) if f.lower().endswith(('.png', '.jpg'))][:5]

    for i, f in enumerate(image_files):
        image_path = os.path.join("dataset/train", f)
        image = cv2.imread(image_path)

        aug_image = image.copy()
        if config.get("augmentation"):
            from data.augmentations import gaussian_noise, brightness, blur
            temp_image = aug_image.astype(np.float32) / 255.0
            if config.get("augmentation").get("gaussian_noise"):
                temp_image = gaussian_noise(temp_image, config.get("augmentation").get("gaussian_std_dev"))
            if config.get("augmentation").get("brightness_adjustment"):
                temp_image = brightness(temp_image, config.get("augmentation").get("brightness_range"))
            if config.get("augmentation").get("blur"):
                temp_image = blur(temp_image, config.get("augmentation").get("blur_type"), config.get("augmentation").get("blur_kernel"))
            aug_image = (temp_image * 255).astype(np.uint8)

        boxes = detect_boxes(aug_image, config)
        labels, confidences = classify_patch(image, boxes, eval_config, model)
        output_image = draw_boxes(image.copy(), boxes, labels, confidences)

        output_path = os.path.join("graded_images", f"{i + 1}.png")
        cv2.imwrite(output_path, output_image)

if __name__ == "__main__":
    main()