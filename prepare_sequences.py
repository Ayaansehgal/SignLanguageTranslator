import os
import numpy as np
import pickle

DATA_DIR = "lsa64_cut"  
VIDEO_EXT = ".mp4"

def load_videos(data_dir):
    sequences = []
    labels = []
    class_names = set()

    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    class_folders.sort()  

    print(f"Found {len(class_folders)} classes")

    label_dict = {label: idx for idx, label in enumerate(class_folders)}

    for label in class_folders:
        class_path = os.path.join(data_dir, label)
        video_files = [f for f in os.listdir(class_path) if f.endswith(VIDEO_EXT)]

        for video_file in video_files:
            video_path = os.path.join(class_path, video_file)
            sequences.append(video_path)
            labels.append(label_dict[label])

    return sequences, labels, label_dict

def main():
    sequences, labels, label_dict = load_videos(DATA_DIR)
    print(f"Processed {len(sequences)} samples")
    print(f"Classes: {label_dict}")

    np.save("sequences.npy", sequences)
    with open("labels.pkl", "wb") as f:
        pickle.dump(labels, f)

if __name__ == "__main__":
    main()
