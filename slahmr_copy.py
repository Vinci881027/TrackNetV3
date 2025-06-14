import os
import shutil
    
match = "241226_1"
tram_path = f"./slahmr_data/{match}/tram_preprocess"
tram_path_init = "/mnt/train-data-7-hdd/vinci/slahmr/slahmr/slahmr/tram_preprocess"
os.makedirs(tram_path, exist_ok=True)

for rally in os.listdir(tram_path_init):
    rally_path = os.path.join(tram_path, rally)
    os.makedirs(rally_path, exist_ok=True)
    data_path = os.path.join(tram_path_init, rally, "tracks.npy")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found.")
    shutil.copy(data_path, os.path.join(rally_path, "tracks.npy"))