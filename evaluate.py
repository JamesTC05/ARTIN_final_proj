import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np # Ensure numpy is imported for fast array math

def evaluate_model():
    # 1. Setup device and transforms
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Load the validation data
    data_dir = "./processed_faces_dataset" 
    dataset = datasets.ImageFolder(root=data_dir, transform=val_transforms)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # 3. Initialize model and load your best weights
    model = models.efficientnet_b0(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    model.load_state_dict(torch.load('best_deepfake_detector.pth', map_location=device))
    model = model.to(device)
    model.eval()

    # Lists to store the true labels and the RAW probabilities
    all_probs = []
    all_labels = []

    print("Running evaluation over dataset (This only happens once!)...")
    
    # 4. Extract raw probabilities
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Extracting Probabilities"):
            inputs = inputs.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(inputs)
            
            # Get the raw percentage (e.g. 0.88) instead of the rounded 0 or 1
            probs = torch.sigmoid(outputs) 
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Convert to numpy arrays for extremely fast math
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # 5. Define the thresholds you want to test
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n" + "="*70)
    print("METRICS ACROSS DIFFERENT THRESHOLDS")
    print("="*70)
    
    # 6. Test all thresholds instantly and print detailed reports
    for thresh in thresholds_to_test:
        # Apply the threshold math here
        preds = (all_probs >= thresh).astype(float)
        
        # Get overall accuracy
        acc = accuracy_score(all_labels, preds)
        
        print(f"\n" + "="*60)
        print(f"THRESHOLD: {thresh:.2f} (Overall Accuracy: {acc * 100:.2f}%)")
        print("="*60)
        
        # Generate and print the detailed classification report
        report = classification_report(
            all_labels, 
            preds, 
            target_names=['Real (0)', 'Fake (1)'], 
            zero_division=0
        )
        print("Detailed Classification Report:")
        print(report)
    
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)

if __name__ == '__main__':
    evaluate_model()