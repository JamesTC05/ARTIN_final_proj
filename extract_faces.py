import os
import cv2
import pandas as pd
from facenet_pytorch import MTCNN
from tqdm import tqdm
import torch

def extract_and_crop_faces(csv_path, base_dir, output_dir, frames_per_video=15, image_size=224):
    # Load the map we made in Step 1
    df = pd.read_csv(csv_path)
    
    # Use GPU if you have one, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize the MTCNN face detector
    # margin ensures we get a bit of the background/neck where deepfake artifacts often appear
    mtcnn = MTCNN(keep_all=False, device=device, image_size=image_size, margin=40)
    
    # Create the output directories for PyTorch/TensorFlow ImageFolder compatibility
    os.makedirs(os.path.join(output_dir, '0_real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, '1_fake'), exist_ok=True)
    
    # Loop through every video in the CSV using tqdm for a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        video_path = os.path.join(base_dir, row['filepath'])
        label = row['label']
        actor_id = row['actor_id']
        video_name = os.path.splitext(row['filename'])[0]
        
        # Determine the correct save folder
        save_folder = '1_fake' if label == 1 else '0_real'
        save_path = os.path.join(output_dir, save_folder)
        
        # Open the video using OpenCV
        v_cap = cv2.VideoCapture(video_path)
        v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if v_len == 0:
            continue
            
        # Calculate exactly which frames to pull so they are evenly spaced
        sample_intervals = [int(i * v_len / frames_per_video) for i in range(frames_per_video)]
        
        frame_count = 0
        for frame_idx in sample_intervals:
            v_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = v_cap.read()
            if not success:
                continue
                
            # OpenCV loads images in BGR format, MTCNN expects RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a unique filename for this specific face image
            face_filename = f"{video_name}_frame{frame_count}.jpg"
            face_filepath = os.path.join(save_path, face_filename)
            
            try:
                # MTCNN detects the face, crops it, resizes it to 224x224, and saves it
                mtcnn(frame, save_path=face_filepath)
            except Exception as e:
                # Sometimes faces aren't detected (e.g., extreme angles), we just skip that frame
                pass
                
            frame_count += 1
            
        v_cap.release()

# --- How to use ---
csv_file = "deepfake_labels.csv"
base_directory = "./"  # The folder containing 'original' and 'manipulated'
output_directory = "./processed_faces_dataset" # Where the faces will be saved

print("Starting extraction... This will take a while!")
extract_and_crop_faces(csv_file, base_directory, output_directory)
print("\nExtraction complete! Your dataset is ready for training.")