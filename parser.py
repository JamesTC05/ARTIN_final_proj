import os
import pandas as pd

def map_split_video_dataset(base_directory):
    dataset_records = []
    
    # Map the folder names to their target labels
    # 0 for Real (original), 1 for Fake (manipulated)
    folders_to_scan = {
        'original': 0,    
        'manipulated': 1  
    }
    
    # Loop through each folder
    for folder_name, label in folders_to_scan.items():
        folder_path = os.path.join(base_directory, folder_name)
        
        # Quick check to ensure the folder exists before proceeding
        if not os.path.exists(folder_path):
            print(f"Warning: Folder not found -> {folder_path}")
            continue
            
        print(f"Scanning '{folder_name}' folder...")
        
        # Loop through all files within the specific folder
        for filename in os.listdir(folder_path):
            if not filename.endswith(".mp4"):
                continue
                
            # Extract actor ID (crucial for ensuring no actor overlap in train/val sets)
            name_no_ext = os.path.splitext(filename)[0]
            parts = name_no_ext.split('__')
            actor_id = parts[0] 
            
            # Store the data, including the full relative path so your dataloader can find it
            dataset_records.append({
                'filename': filename,
                'actor_id': actor_id,
                'label': label,
                'source_folder': folder_name,
                'filepath': os.path.join(folder_path, filename)
            })
            
    # Convert to a pandas DataFrame
    df = pd.DataFrame(dataset_records)
    return df

# --- How to run it ---

# Since parser.py is in the same directory as the folders, we use the current directory "./"
base_dir = "./" 

# Generate the dataframe
df_labels = map_split_video_dataset(base_dir)

# Save to CSV
csv_filename = "deepfake_labels.csv"
df_labels.to_csv(csv_filename, index=False)
print(f"\nSuccess! Found {len(df_labels)} total videos. Saved mapped data to {csv_filename}")