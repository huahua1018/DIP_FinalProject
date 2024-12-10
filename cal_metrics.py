import numpy as np
from pytorch_msssim import ssim
from PIL import Image
import csv
import argparse
import os
import re
import torch
from tqdm.auto import tqdm

def psnr(img_orig, img_out):
    squared_error = np.square(img_orig - img_out)
    mse = np.mean(squared_error)
    psnr = 10 * np.log10(255.0 / mse)
    return psnr

def calculate_psnr_ssim(images_original, images_reconstructed):

    psnr_list = []
    ssim_list = []
    for img_orig, img_recon in tqdm(zip(images_original, images_reconstructed)):
        # Calculate PSNR
        psnr_val = psnr(img_orig, img_recon)
        psnr_list.append(psnr_val)
        # Calculate SSIM
        img_orig_tensor = torch.tensor(img_orig).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, height, width)
        img_recon_tensor = torch.tensor(img_recon).permute(2, 0, 1).unsqueeze(0).float()
        diff = img_orig_tensor - img_recon_tensor
        ssim_val = ssim(img_orig_tensor , img_recon_tensor, data_range=255.0).item()
        ssim_list.append(ssim_val)

    metrics = {
        "mean_psnr": np.mean(psnr_list),
        "mean_ssim": np.mean(ssim_list),
        "psnr_list": psnr_list,
        "ssim_list": ssim_list
    }
    return metrics

def split_and_sort(root_gt_path, root_recon_path, filename1, filename2):
    if not os.path.exists(root_gt_path) or not os.path.exists(root_recon_path):
        raise FileNotFoundError(f"Directory {root_gt_path} or {root_recon_path} not exist")

    gt_files = [f for f in os.listdir(root_gt_path) 
        if os.path.isfile(os.path.join(root_gt_path, f)) and f.endswith('.png')]
    recon_files = [f for f in os.listdir(root_recon_path) 
        if os.path.isfile(os.path.join(root_recon_path, f)) and f.endswith('.png')]
    files = gt_files + recon_files
    group1 = []
    group2 = []

    def extract_number(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(0)) if match else float('inf')
    for f in files:
        if filename1(f):
            file_path = os.path.join(root_gt_path, f)
        else:
            file_path = os.path.join(root_recon_path, f)
        img = Image.open(file_path)
        img_array = np.array(img)
        if filename1(f):
            group1.append((f, img_array))
        elif filename2(f):
            group2.append((f, img_array))

    group1.sort(key=lambda x: extract_number(x[0]))
    group2.sort(key=lambda x: extract_number(x[0]))
    group1_arrays = [img_array for _, img_array in group1]
    group2_arrays = [img_array for _, img_array in group2]
    return group1_arrays, group2_arrays

def save_metrics_to_csv(metrics, filename="metrics_list.csv"):
    with open(filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Image Index", "PSNR", "SSIM"])  # Write header
        for idx, (psnr, ssim) in enumerate(zip(metrics["psnr_list"], metrics["ssim_list"])):
            writer.writerow([idx + 1, psnr, ssim])
        writer.writerow(["Mean PSNR", metrics["mean_psnr"]])
        writer.writerow(["Mean SSIM", metrics["mean_ssim"]])



parser = argparse.ArgumentParser()
parser.add_argument('--root_gt', '-r_gt', default='./reflection-diffusion-pytorch/results_out_0.4_l2_500', type = str)         #TODO: change this to gt image dir path
parser.add_argument('--root_recon', '-r_recon', default='./reflection-diffusion-pytorch/results_out_0.4_l2_500', type = str)   #TODO: change this to recon image dir path
parser.add_argument('--direct', '-d', default=False, type = bool)
parser.add_argument('--output_filename', '-o', default='./reflection-diffusion-pytorch/results_out_0.4_l2_500/metrics_list.csv', type = str)                        #TODO: change this to output path + filename
args = parser.parse_args()
root_gt_path = args.root_gt
root_recon_path = args.root_recon

condition1 = lambda filename: "sample-gt-" in filename.lower()                  #TODO: change this to filenname format of gt-image
if not args.direct:
    condition2 = lambda filename: "sample-x0-" in filename.lower()           #TODO: change this to filenname format of reconstructed image
else:
    condition2 = lambda filename: "sample-direct-" in filename.lower()   #TODO: change this to filenname format of direct reconstructed image

images_original, images_reconstructed = split_and_sort(root_gt_path, root_recon_path, condition1, condition2)
metrics = calculate_psnr_ssim(images_original, images_reconstructed)

print(f"Mean PSNR: {metrics['mean_psnr']}")
print(f"Mean SSIM: {metrics['mean_ssim']}")
save_metrics_to_csv(metrics, args.output_filename)