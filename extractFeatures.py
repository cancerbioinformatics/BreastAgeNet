import os
import glob
import random
import torch
import argparse
import pandas as pd
from utils_features import *



def Extract_features_from_WSIs(root, model, transform, stainFunc, batch_size, num_workers, device):
    """Extract features from WSIs (e.g., '.ndpi' files)."""
    
    wsinames = os.listdir(f'{root}/WSIs')
    for wsi_id in wsinames:
        output_dir = f"{root}/FEATUREs/{wsi_id}"
        fname = f"{output_dir}/{wsi_id}_bagFeature_{model_name}_{stainFunc}.h5"
        
        if not os.path.exists(fname):
            print(f"Processing WSI: {wsi_id}", flush=True)
            file = glob.glob(f"{root}/FEATUREs/{wsi_id}/{wsi_id}*_TC_512_patch_wsi.csv")
            
            if file:
                print(f"Found CSV file: {file[0]}")
                bag_df = pd.read_csv(file[0])
                print(f"Number of patches: {len(bag_df)}")
                
                if len(bag_df) > 0:
                    try:
                        bag_dataset = Dataset_fromWSI(bag_df, WSIs, stainFunc, transforms_eval=transform)
                        extract_features(model, bag_dataset, batch_size, num_workers, device, fname)
                    except:
                        continue




def Extract_features_from_patches(root, patch_csv, model, transform, stainFunc, batch_size, num_workers, device):
    """Extract features from individual patch images (e.g., '.png' files)."""
    
    df = pd.read_csv(patch_csv) # "/scratch_tmp/prj/cb_normalbreast/prj_NBTClassifier/TC512_externaltesting_EPFL.csv"
    df["wsi_id"] = df["file_path"].apply(lambda x: os.path.basename(x).split("_HE")[0])
    df["patch_id"] = df["file_path"].apply(lambda x: os.path.basename(x).split(".png")[0])
    print(f"Total patches: {len(df)}")
    
    fname = f"{root}/FEATUREs/{wsi_id}/{wsi_id}_bagFeature_{model_name}_{stainFunc}.h5"
    for wsi_id, bag_df in df.groupby("wsi_id"):
        print(f"Processing WSI: {wsi_id}", flush=True)
        try:
            bag_dataset = Dataset_frompatch(bag_df, stainFunc, transform)
            extract_features(model, bag_dataset, batch_size, num_workers, device, fname)
        except:
            continue



parser = argparse.ArgumentParser(description="Feature Extraction for BreastAgeNet")
parser.add_argument("--model", type=str, default="UNI", help="Model name (e.g., 'UNI', 'ResNet50', 'gigapath', 'phikon')")
parser.add_argument("--stain", type=str, default="augmentation", help="Staining function (e.g., 'augmentation', 'reinhard')")
parser.add_argument("--image_type", type=str, choices=["WSI", "patch"], default="WSI", help="Input type: 'WSI' or 'patch'")
parser.add_argument("--IMAGES", type=str, default="./project-data/WSIs", help="Path to WSI directory")
parser.add_argument("--patch_csv", type=str, default="", help="when image_type is patch, patch_csv provides their tissue classification results")
parser.add_argument("--FEATURES", type=str, default="./project-data/FEATURES", help="Path to features directory")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size for feature extraction")
parser.add_argument("--num_workers", type=int, default=2, help="Number of workers for data loading")
args = parser.parse_args()


model_name = args.model
stainFunc = args.stain
image_type = args.image_type
IMAGES = args.IMAGES
patch_csv = args.patch_csv
FEATURES = args.FEATURES
batch_size = args.batch_size
num_workers = args.num_workers


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}", flush=True)
model, transform = get_model(model_name, device)
print(f"Loaded Model: {model_name} with Stain Function: {stainFunc}", flush=True)


if image_type == 'WSI':
    Extract_features_from_WSIs(IMAGES, model, transform, stainFunc, batch_size, num_workers, device)
elif image_type == 'patch':
    Extract_features_from_patches(IMAGES, args.patch_csv, model, transform, stainFunc, batch_size, num_workers, device)
