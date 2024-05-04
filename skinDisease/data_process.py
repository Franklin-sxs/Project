import os
import pandas as pd
from shutil import copy2
from sklearn.model_selection import train_test_split


metadata_path = 'dataset/HAM10000_metadata.csv'
images_path = 'dataset/HAM10000_images'
base_output_path = 'dataset/HAM10000'

metadata_df = pd.read_csv(metadata_path)

train_df, temp_df = train_test_split(metadata_df, test_size=0.2, stratify=metadata_df['dx'])
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['dx'])


os.makedirs(base_output_path, exist_ok=True)


def copy_images(df, subset_name):
    subset_path = os.path.join(base_output_path, subset_name)
    os.makedirs(subset_path, exist_ok=True)

    for _, row in df.iterrows():
        class_dir = os.path.join(subset_path, row['dx'])
        os.makedirs(class_dir, exist_ok=True)

        src_image_path = os.path.join(images_path, row['image_id'] + '.jpg')
        dst_image_path = os.path.join(class_dir, row['image_id'] + '.jpg')
        copy2(src_image_path, dst_image_path)


copy_images(train_df, 'train')
copy_images(val_df, 'val')
copy_images(test_df, 'test')

print("Dataset splitting completed.")