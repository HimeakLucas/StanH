import random
import os
from pathlib import Path
import shutil
import argparse

def split(input_dir, output_dir, train_perc=70, test_perc=20, validation_perc=10, seed=42,files_limit=11430):

    random.seed(seed)
    p = Path(input_dir)
    if (train_perc + validation_perc + test_perc) != 100:
        print("Error: the sum of the split percentages must be equals to 100")
        return

    if not p.is_dir():
        print("Error: directory not found")
        return

    path_train = Path(output_dir) /"train"/"data"
    path_val = Path(output_dir) /"val"/"data"
    path_test = Path(output_dir) /"test"/"data"
    
    path_train.mkdir(parents=True, exist_ok=True)
    path_val.mkdir(parents=True, exist_ok=True)
    path_test.mkdir(parents=True, exist_ok=True)

    extensions = ('.png', '.jpg', '.jpeg', '.bpm', '.tif')

    images = [] #list of paths
    for file in os.listdir(input_dir):
        if file.lower().endswith(extensions):
            images.append(file)
            if len(images) >= files_limit: break

    random.shuffle(images)
    total = len(images)
    train_idx = int(total * (train_perc / 100))
    val_idx = train_idx + int(total * (validation_perc / 100))

    train_files = images[:train_idx]
    val_files = images[train_idx:val_idx]
    test_files = images[val_idx:]

    def copy_files(files, source, dest):
        for f in files:
            shutil.copy2(Path(source)/f , Path(dest)/f)
        print(f"Copied {len(files)} files from {source}")

    copy_files(train_files, input_dir, path_train)
    copy_files(val_files, input_dir, path_val)
    copy_files(test_files, input_dir, path_test)


if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Split dataset into train, validation and test sets.")
    argparser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images.")
    argparser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory where the split datasets will be stored.")
    argparser.add_argument("--train_perc", type=int, default=70, help="Percentage of images to be used for training (default: 70).")
    argparser.add_argument("--test_perc", type=int, default=20, help="Percentage of images to be used for testing (default: 20).")
    argparser.add_argument("--validation_perc", type=int, default=10, help="Percentage of images to be used for validation (default: 10).")
    argparser.add_argument("--files_limit", type=int, default=11430, help="Limit for the total number of images of the dataset (default: 10400)")
    argparser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling the dataset (default: 42).") 

    args = argparser.parse_args()
    split(args.input_dir, args.output_dir, args.train_perc, args.test_perc, args.validation_perc, args.seed, args.files_limit)
