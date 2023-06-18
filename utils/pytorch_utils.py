from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch
from PIL import Image
import os

default_transform=transforms.Compose([transforms.Resize(320), transforms.ToTensor()])

class XRayDataset(Dataset):
    def __init__(self, dataframe, labels, image_dir, transforms=default_transform):
        self.dataframe = dataframe
        self.labels = labels
        self.image_dir = image_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_id = row["Image"]
        image_path = os.path.join(self.image_dir, image_id)
        image = Image.open(image_path)
        if self.transforms:
            image = self.transforms(image)
        labels = row[self.labels]
        return (
            image,
            torch.tensor(labels),
        )

def process_imagenet_predictions(preds):
    with open("./ressources-atelier-ia-medical/data/imagenet_classes.txt", "r") as f:
        img_classes = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(preds[0], 5)
    for id,prob in zip(top5_catid,top5_prob):
      print(f'Label prédit: {img_classes[id]}: {prob * 100:.4f} %')
