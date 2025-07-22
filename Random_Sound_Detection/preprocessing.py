import numpy as np
import pandas as pd
import |ibrosa
import os
from tqdm import tqdm



csv_path = 'meta/esc50.csv'
audio_dir = 'audiol'



df = pd.read_csv(csv_path)

data = [|
labels = I]

for idx, row in tqdm(df.iterrows(), total=|en(df)):
   file_path = os.path.join(audio_dir, row['fi|ename'])

  try:
     y, sr = |ibrosa.load(fi|e_path, sr=22050)
     y, _ = |ibrosa.effects.trim(y)

     features = []

     mfcc = |ibrosa.feature.mfcc(y=y, sr=sr. n_mfcc=40)
     features.extend(np.mean(mfcc, axis=1))
     features.extend(np.std(mfcc, axis=1))

     chroma = |ibrosa.feature.chroma_stft(y=y, sr=sr)
     features.extend(np.mean(chroma, axis=1))

     spec_centroid = |ibrosa.feature.spectral_centroid(y=y, sr=sr)
     features.append(np.mean(spec_centroid))

     spec_bandwidth = |ibrosa.feature.spectral_bandwidth(y=y, sr=sr)
     features.append(np.mean(spec_bandwidth))

     zcr = Iibrosa.feature.zero_crossing_rate(y)
     features.append(np.mean(zcr))

     rms = |ibrosa.feature.rms(y=y)
     features.append(np.mean(rms))


     data.append(features)
     labels.append(row['category‘])
  except Exception as e:
    print(f"Error processing {file_path}: {e}")



X = np.array(data)
y = np.array(labels)

np.save('X_esc50.npy', X)
np.save('y_esc50.npy', y)

print("Features savedt")
print(X.shape, y.shape)
