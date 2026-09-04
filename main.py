import os
import re
import cv2
import random
import copy
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import warnings

warnings.filterwarnings("ignore")

def set_global_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_global_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = '/root/autodl-tmp/V2_experiment'
RAW_DIR = os.path.join(BASE_DIR, 'raw_images')
TILES_DIR = os.path.join(BASE_DIR, 'tiles')
CSV_TRAIN = os.path.join(BASE_DIR, 'train_isolated_enriched.csv')
CSV_VAL = os.path.join(BASE_DIR, 'val_isolated_enriched.csv')

if not os.path.exists(TILES_DIR):
    os.makedirs(TILES_DIR)

def calc_13_features(pil_img, x_coord=0, y_coord=0):
    img_arr = np.array(pil_img)
    r, g, b = img_arr[:,:,0], img_arr[:,:,1], img_arr[:,:,2]
    gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

    r_m, g_m, b_m = np.mean(r), np.mean(g), np.mean(b)
    r_v, g_v, b_v = np.var(r), np.var(g), np.var(b)
    gray_v = np.var(gray)
    brightness = np.mean(gray)

    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1] + 1e-6)

    hist = np.histogram(gray, bins=256, range=(0, 256))[0]
    p = hist / np.sum(hist)
    p = p[p > 0]
    entropy = -np.sum(p * np.log2(p)) if len(p) > 0 else 0.0

    spatial_x = x_coord / 1024.0
    spatial_y = y_coord / 1024.0
    aspect_ratio = 1.0

    return [r_m, g_m, b_m, r_v, g_v, b_v, gray_v, brightness,
            edge_density, entropy, spatial_x, spatial_y, aspect_ratio]

def process_and_slice():
    data_records = []
    tile_size, stride = 200, 100
    image_list = []

    for label, folder in [(1, 'banksy'), (0, 'not_banksy')]:
        folder_path = os.path.join(RAW_DIR, folder)
        if os.path.exists(folder_path):
            for img_name in os.listdir(folder_path):
                if not img_name.startswith('.'):
                    image_list.append((os.path.join(folder_path, img_name), img_name, label))

    for img_path, img_name, label in tqdm(image_list):
        try:
            img = Image.open(img_path).convert('RGB')
            img.thumbnail((1024, 1024))
            w, h = img.size
            parent_id = os.path.splitext(img_name)[0]

            for y in range(0, h - tile_size + 1, stride):
                for x in range(0, w - tile_size + 1, stride):
                    tile = img.crop((x, y, x + tile_size, y + tile_size))
                    features = calc_13_features(tile, x, y)
                    
                    if features[9] >= 2.0:
                        tile_filename = f"{parent_id}_x{x}_y{y}.jpg"
                        tile.save(os.path.join(TILES_DIR, tile_filename))
                        data_records.append([tile_filename, parent_id] + features + [label])
        except Exception:
            pass

    columns = ['filename', 'parent_id', 'r_m', 'g_m', 'b_m', 'r_v', 'g_v', 'b_v', 'gray_v', 'brightness',
               'edge_density', 'entropy', 'spatial_x', 'spatial_y', 'aspect_ratio', 'label']
    df = pd.DataFrame(data_records, columns=columns)

    unique_parents = df['parent_id'].unique()
    train_parents, val_parents = train_test_split(unique_parents, test_size=0.2, random_state=42)

    df_train = df[df['parent_id'].isin(train_parents)].drop(columns=['parent_id'])
    df_val = df[df['parent_id'].isin(val_parents)].drop(columns=['parent_id'])

    df_train.to_csv(CSV_TRAIN, index=False)
    df_val.to_csv(CSV_VAL, index=False)

    return df_train, df_val

class StrictDataset(Dataset):
    def __init__(self, csv_file, img_dirs, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dirs = img_dirs if isinstance(img_dirs, list) else [img_dirs]
        self.transform = transform
        self.img_col = 'filename' if 'filename' in self.df.columns else self.df.columns[0]
        self.feature_cols = [c for c in self.df.columns if c not in [self.img_col, 'label']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        
        img_path = None
        for d in self.img_dirs:
            p = os.path.join(d, img_name)
            if os.path.exists(p): 
                img_path = p
                break
        if img_path is None:
            for d in self.img_dirs:
                p = os.path.join(d, img_name + '.jpg')
                if os.path.exists(p): 
                    img_path = p
                    break
        if img_path is None: 
            raise FileNotFoundError(f"Missing file: {img_name}")
            
        image = Image.open(img_path).convert('RGB')
        if self.transform: 
            image = self.transform(image)
        
        csv_features = row[self.feature_cols].values.astype(np.float32)
        return image, torch.tensor(csv_features), int(row['label'])

class PretrainedVisionNet(nn.Module):
    def __init__(self, num_classes=2):
        super(PretrainedVisionNet, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(512, num_classes))
        
    def forward(self, x): 
        return self.fc(self.features(x))

class ResNetMultiModalNet(nn.Module):
    def __init__(self, num_csv_features, num_classes=2, mode='fusion'):
        super(ResNetMultiModalNet, self).__init__()
        self.mode = mode
        self.cnn = PretrainedVisionNet(num_classes)
        self.csv_mlp = nn.Sequential(
            nn.Linear(num_csv_features, 128), nn.ReLU(), nn.BatchNorm1d(128), 
            nn.Dropout(0.3), nn.Linear(128, num_classes)
        )
        
    def forward(self, image_tensor, csv_tensor):
        if self.mode == 'fusion': 
            return self.cnn(image_tensor) + self.csv_mlp(csv_tensor)
        elif self.mode == 'image_only': 
            return self.cnn(image_tensor)
        elif self.mode == 'csv_only': 
            return self.csv_mlp(csv_tensor)

def run_resnet_experiment(mode_name, train_loader, val_loader, num_epochs=20, device='cuda'):
    model = ResNetMultiModalNet(num_csv_features=13, num_classes=2, mode=mode_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    history_val_acc = []
    best_acc = 0.0
    best_weights = None
    
    for epoch in range(num_epochs):
        model.train()
        for images, csv_data, labels in train_loader:
            images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images, csv_data), labels)
            loss.backward()
            optimizer.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, csv_data, labels in val_loader:
                images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)
                outputs = model(images, csv_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        val_acc = 100 * correct / total if total > 0 else 0
        history_val_acc.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            
        scheduler.step()
        
    return history_val_acc, best_weights, best_acc

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs_to_run = 20
    set_global_seed(42)
    
    if not (os.path.exists(CSV_TRAIN) and os.path.exists(CSV_VAL)):
        process_and_slice()
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),                          
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    search_dirs = [TILES_DIR, '/root/autodl-tmp/train_slices_b', '/root/autodl-tmp/val_slices_b']
    train_dataset = StrictDataset(CSV_TRAIN, search_dirs, transform_train)
    val_dataset = StrictDataset(CSV_VAL, search_dirs, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    acc_img_lw, _, _ = run_resnet_experiment('image_only', train_loader, val_loader, epochs_to_run, device)
    acc_csv_lw, _, _ = run_resnet_experiment('csv_only', train_loader, val_loader, epochs_to_run, device)
    acc_fusion_lw, best_fusion_weights, best_fusion_acc = run_resnet_experiment('fusion', train_loader, val_loader, epochs_to_run, device)

    eval_model = ResNetMultiModalNet(num_csv_features=13, num_classes=2, mode='fusion').to(device)
    eval_model.load_state_dict(best_fusion_weights)
    eval_model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, csv_data, labels in val_loader:
            images, csv_data, labels = images.to(device), csv_data.to(device), labels.to(device)
            outputs = eval_model(images, csv_data)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    print(classification_report(all_labels, all_preds))
