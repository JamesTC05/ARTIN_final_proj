import torch
import cv2
from torchvision import models, transforms
import torch.nn as nn
from facenet_pytorch import MTCNN
from PIL import Image

def predict_video(video_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")

    # 1. Rebuild the Exact Model Architecture
    model = models.efficientnet_b0(weights=None) # We don't need pretrained weights from the internet anymore
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, 1)
    
    # 2. Load Your Trained Weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval() # CRITICAL: Set to evaluation mode!

    # 3. Setup the Face Detector and Image Transforms
    mtcnn = MTCNN(keep_all=False, device=device, image_size=224, margin=40)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 4. Open the Target Video
    v_cap = cv2.VideoCapture(video_path)
    v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if v_len == 0:
        return "Could not read video."

    print(f"Analyzing {video_path}...")
    
    # We don't need to check every single frame, checking 15 evenly spaced frames is usually enough
    frames_to_check = 15
    sample_intervals = [int(i * v_len / frames_to_check) for i in range(frames_to_check)]
    
    predictions = []

    # 5. Process the Video Frame by Frame
    for frame_idx in sample_intervals:
        v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = v_cap.read()
        if not success:
            continue
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and crop the face
        face = mtcnn(frame)
        
        if face is not None:
            # The face tensor is already normalized between -1 and 1 by MTCNN, 
            # but we need it in the format our model expects.
            # Convert back to PIL Image temporarily to apply our specific PyTorch transforms
            face_img = transforms.ToPILImage()(face * 0.5 + 0.5) 
            face_tensor = transform(face_img).unsqueeze(0).to(device) # Add batch dimension

            # Make the prediction!
            with torch.no_grad(): # Don't track gradients (saves memory)
                output = model(face_tensor)
                probability = torch.sigmoid(output).item()
                predictions.append(probability)

    v_cap.release()

    # 6. Calculate Final Verdict
    if not predictions:
        return "No faces detected in this video."

    # Average the probabilities across all analyzed frames
    avg_probability = sum(predictions) / len(predictions)
    
    print("-" * 30)
    print(f"Average Fake Probability: {avg_probability * 100:.2f}%")
    
    if avg_probability >= 0.5:
        print("Verdict: 🚨 MANIPULATED (FAKE) 🚨")
    else:
        print("Verdict: ✅ AUTHENTIC (REAL) ✅")
    print("-" * 30)

# --- How to use ---
if __name__ == '__main__':
    # Point this to your saved weights file
    saved_model = "best_deepfake_detector.pth" 
    
    # Point this to ANY mp4 file you want to test!
    target_video = "./original/07__kitchen_pan.mp4" 
    
    predict_video(target_video, saved_model)