import os
import numpy as np
import librosa

DATASET_PATH = "data/labeled"

EMOTIONS = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "fear"
]

N_MFCC = 40

MAX_PAD_LEN = 173

emotion_to_label = {
    "angry": 0,
    "happy": 1,
    "sad": 2,
    "neutral": 3,
    "fear": 4
}


def extract_mfcc(file_path):

    try:

        audio, sample_rate = librosa.load(
            file_path,
            sr=16000
        )

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=N_MFCC
        )

        mfcc = mfcc.T


        if mfcc.shape[0] < MAX_PAD_LEN:

            pad_width = MAX_PAD_LEN - mfcc.shape[0]

            mfcc = np.pad(
                mfcc,
                pad_width=((0, pad_width), (0, 0)),
                mode='constant'
            )

        else:

            mfcc = mfcc[:MAX_PAD_LEN]

        return mfcc

    except Exception as e:

        print(f"Error processing {file_path}")
        print(e)

        return None

X = []
y = []

print("\nExtracting MFCC features...\n")

for emotion in EMOTIONS:

    emotion_folder = os.path.join(
        DATASET_PATH,
        emotion
    )

    if not os.path.exists(emotion_folder):
        continue

    files = os.listdir(emotion_folder)

    print(f"{emotion}: {len(files)} files")

    for file_name in files:

        if file_name.endswith(".wav"):

            file_path = os.path.join(
                emotion_folder,
                file_name
            )

            mfcc = extract_mfcc(file_path)

            if mfcc is not None:

                X.append(mfcc)

                y.append(
                    emotion_to_label[emotion]
                )


X = np.array(X)
y = np.array(y)


print("\n===================================")
print("FEATURE EXTRACTION COMPLETE")
print("===================================")

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

os.makedirs("data", exist_ok=True)

np.save("data/X.npy", X)
np.save("data/y.npy", y)

print("\nDataset saved successfully!")