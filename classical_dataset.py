import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm  

classical_folder = os.path.expanduser("~/Desktop/Personal-Projects/algo-music-analysis/gtzan-dataset-music-genre-classification/versions/1/Data/genres_original/classical")

data = []
keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# go through each audio clip in the classical dataset
for file in tqdm(os.listdir(classical_folder)):
    file_path = os.path.join(classical_folder, file)

    # load audio and extract features
    y, sr = librosa.load(file_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    key_strength = chroma.sum(axis=1)
    key_index = np.argmax(key_strength)
    predicted_key = keys[key_index]
    
    data.append({
        "File": file,
        "Key": predicted_key
    })

# save data to csv file and output the saved path
df = pd.DataFrame(data)

output_path = os.path.expanduser("~/Desktop/Personal-Projects/algo-music-analysis/classical_keys_analysis.csv")
df.to_csv(output_path, index=False)

print(f"Saved path: {output_path}.")
