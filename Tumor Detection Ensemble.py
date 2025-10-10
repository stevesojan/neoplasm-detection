# Imports
import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from torchvision.models import resnet18, ResNet18_Weights

# --- 1. Dataset Definition ---
class MammogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for class_folder in os.listdir(self.root_dir):
            class_path = os.path.join(self.root_dir, class_folder)
            if not os.path.isdir(class_path):
                continue  # Skip .zip or any non-directory

            label = 1 if 'MALIGNANT' in class_folder.upper() else 0
            inner_class_path = os.path.join(class_path, os.listdir(class_path)[0])  # 'Benign' or 'Malignant' folder

            for img_file in os.listdir(inner_class_path):
                img_path = os.path.join(inner_class_path, img_file)
                if 'CC' in img_file.upper():
                    side = 'L' if 'LEFT' in img_file.upper() else 'R'
                    match_mlo = [m for m in os.listdir(inner_class_path) if 'MLO' in m.upper() and side in m.upper()]
                    if match_mlo:
                        samples.append({
                            'cc': os.path.join(inner_class_path, img_file),
                            'mlo': os.path.join(inner_class_path, match_mlo[0]),
                            'label': label
                        })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cc_img = Image.open(sample['cc']).convert('RGB')
        mlo_img = Image.open(sample['mlo']).convert('RGB')
        label = sample['label']

        if self.transform:
            cc_img = self.transform(cc_img)
            mlo_img = self.transform(mlo_img)

        return cc_img, mlo_img, torch.tensor(label, dtype=torch.long)

# --- 2. Create Bootstrapped Subsets ---
def create_bootstrap_datasets(dataset, num_samples):
    indices = [random.randint(0, len(dataset)-1) for _ in range(num_samples)]
    return Subset(dataset, indices)

# --- 3. Model Definitions ---
class BaseEnsembleModel(nn.Module):
    def __init__(self, feature_extractor, feature_dim):
        super().__init__()
        self.cnn = feature_extractor
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 2)
        )

    def forward(self, cc, mlo):
        cc_feat = self.cnn(cc)
        mlo_feat = self.cnn(mlo)
        combined = torch.cat([cc_feat, mlo_feat], dim=1)
        return self.fc(combined)

# --- 4. Feature Extractors ---
def get_shufflenet():
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = shufflenet_v2_x0_5(weights=weights)
    model.fc = nn.Identity()
    return nn.Sequential(model, nn.Flatten())

def get_mobilenet_v2():
    weights = MobileNet_V2_Weights.DEFAULT
    model = mobilenet_v2(weights=weights)
    model.classifier = nn.Identity()  # Remove classifier head
    return nn.Sequential(model, nn.Flatten())

def get_resnet18():
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    return nn.Sequential(model, nn.Flatten())



# --- 5. Ensemble Prediction Function ---
def ensemble_predict(models, cc, mlo):
    outputs = [F.softmax(model(cc, mlo), dim=1) for model in models]
    avg_output = torch.stack(outputs).mean(dim=0)
    return avg_output

# --- 6. Training and Evaluation ---
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0
    for cc, mlo, labels in dataloader:
        cc, mlo, labels = cc.to(device), mlo.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(cc, mlo)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate(models, dataloader, device):
    all_preds, all_labels = [], []
    for cc, mlo, labels in dataloader:
        cc, mlo = cc.to(device), mlo.to(device)
        with torch.no_grad():
            outputs = ensemble_predict(models, cc, mlo)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds)
        all_labels.extend(labels)
    acc = accuracy_score(all_labels, all_preds)
    return acc

def show_predictions(models, val_set, device, class_names=['Benign', 'Malignant']):
    print("\n--- Image + Prediction Visualization ---")
    for i in range(5):  # Change 5 to more if you want
        cc, mlo, label = val_set[i]
        cc_tensor = cc.unsqueeze(0).to(device)
        mlo_tensor = mlo.unsqueeze(0).to(device)

        with torch.no_grad():
            output = ensemble_predict(models, cc_tensor, mlo_tensor)
            pred = torch.argmax(output, dim=1).item()

        # Convert tensors to images
        cc_img = cc.permute(1, 2, 0).cpu().numpy()
        mlo_img = mlo.permute(1, 2, 0).cpu().numpy()

        # Rescale from normalized [-1,1] to [0,1]
        cc_img = (cc_img * 0.5 + 0.5).clip(0, 1)
        mlo_img = (mlo_img * 0.5 + 0.5).clip(0, 1)

        # Plot side by side
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        axs[0].imshow(cc_img)
        axs[0].set_title("CC View")
        axs[1].imshow(mlo_img)
        axs[1].set_title("MLO View")

        for ax in axs:
            ax.axis('off')

        plt.suptitle(f"Prediction: {class_names[pred]} | Ground Truth: {class_names[label]}", fontsize=12)
        plt.tight_layout()
        plt.show()

# --- 7. Putting It All Together ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = r'/\Dataset'  # <-- Change this

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),  # Flip left/right
        transforms.RandomRotation(10),  # Small rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
    ])

    full_dataset = MammogramDataset(dataset_path, transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    boot_resnet = create_bootstrap_datasets(train_set, train_size)
    boot_densenet = create_bootstrap_datasets(train_set, train_size)
    boot_resnet2 = create_bootstrap_datasets(train_set, train_size)
    loader_resnet2 = DataLoader(boot_resnet2, batch_size=16, shuffle=True)

    loader_resnet = DataLoader(boot_resnet, batch_size=16, shuffle=True)
    loader_densenet = DataLoader(boot_densenet, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=False)

    resnet_model = BaseEnsembleModel(get_shufflenet(), 1024).to(device)
    mobilenet_model = BaseEnsembleModel(get_mobilenet_v2(), 1280).to(device)
    resnet_model_2 = BaseEnsembleModel(get_resnet18(), 512).to(device)

    opt_r = torch.optim.Adam(resnet_model.parameters(), lr=1e-4, weight_decay=1e-5)
    opt_m = torch.optim.Adam(mobilenet_model.parameters(), lr=1e-4, weight_decay=1e-5)
    opt_r2 = torch.optim.Adam(resnet_model_2.parameters(), lr=1e-4, weight_decay=1e-5)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(10):
        loss_r = train(resnet_model, loader_resnet, opt_r, criterion, device)
        loss_m = train(mobilenet_model, loader_densenet, opt_m, criterion, device)
        loss_r2 = train(resnet_model_2, loader_resnet2, opt_r2, criterion, device)
        acc = evaluate([resnet_model, mobilenet_model, resnet_model_2], val_loader, device)
        print(f"Epoch {epoch + 1}: ShuffleNet Loss={loss_r:.4f}, MobileNet Loss={loss_m:.4f},ResNet Loss={loss_r2:.4f}, Ensemble Acc={acc:.4f}")

    # After training
    print("\n--- Prediction Sample Output ---")
    class_names = ['Benign', 'Malignant']
    model_ensemble = [resnet_model, mobilenet_model, resnet_model_2]

    # Run on first 10 samples of validation set
    for i in range(10):
        cc, mlo, label = val_set[i]
        cc = cc.unsqueeze(0).to(device)
        mlo = mlo.unsqueeze(0).to(device)

        with torch.no_grad():
            output = ensemble_predict(model_ensemble, cc, mlo)
            pred = torch.argmax(output, dim=1).item()

        print(f"Sample {i + 1}: Prediction = {class_names[pred]}, Ground Truth = {class_names[label]}")


    show_predictions([resnet_model, mobilenet_model], val_set, device)


from collections import Counter
print(Counter([s['label'] for s in full_dataset.samples]))

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# Gather all true labels and predictions
all_preds, all_labels = [], []
for cc, mlo, labels in val_loader:
    cc, mlo = cc.to(device), mlo.to(device)
    with torch.no_grad():
        outputs = ensemble_predict(model_ensemble, cc, mlo)
    preds = torch.argmax(outputs, dim=1).cpu()
    all_preds.extend(preds)
    all_labels.extend(labels)

# Generate confusion matrix
cm = confusion_matrix(all_labels, all_preds)
print("\nConfusion Matrix:")
print(cm)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# Classification report
print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=['Benign', 'Malignant']))
