# ARTIN_final_proj

https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset/data?select=DFD_original+sequences <br>
download dataset and rename folders to "manipulated" and "original" <br>
<img width="307" height="145" alt="image" src="https://github.com/user-attachments/assets/5b4a100b-b0b3-43d1-a7db-d939a48dd58a" /> <br>


pip install opencv-python facenet-pytorch pandas tqdm <br>
pip install torchvision <br>

pytorch with CUDA (do this if have nvidia GPU) <br>
pip uninstall torch torchvision torchaudio (uninstall because if u did pip install torchvision automatically CPU only) <br>
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 //cu128(depends on gpu) - rtx 5060 <br>

Run the ff in order:
parser
extract_faces
model_setup
inf

In inf line 93  target_video = "./test/tom_cruise_fake.mp4" #change this to any vid
