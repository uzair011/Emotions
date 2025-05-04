import torch
from torch.utils.data import Dataset
from torchvision import transforms

class FER2013Dataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = []
        self.labels = []
        self.transform = transform

        # Read the CSV file
        with open(csv_file, 'r') as f:
            next(f)  # Skip header
            for line in f:
                pixels, label = line.split(",")[1:3]
                pixels = torch.tensor([int(p) for p in pixels.split()], dtype=torch.float32).view(48, 48)
                label = int(label.strip())
                self.data.append(pixels)
                self.labels.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx], self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Define reusable transforms
def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])
