## Standard libraries
import os
import json
import math
import random
import time
import numpy as np
import scipy.linalg
from PIL import Image, ImageFilter  

#Libraries for getting dataset
import urllib.request
from urllib.error import HTTPError
import zipfile

## Progress bar
from tqdm.notebook import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
# Torchvision
import torchvision
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms import v2
# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
#local libs
from models import Generator

## Imports for plotting
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('svg', 'pdf') # For export
from matplotlib.colors import to_rgb
import matplotlib
matplotlib.rcParams['lines.linewidth'] = 2.0
import seaborn as sns
sns.set()


# Path to the folder where the datasets are/should be downloaded (e.g. MNIST)
DATASET_PATH = "./data"
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = "./saved_models/tutorial10"

# Setting the ]seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Fetching the device that will be used throughout this notebook
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
#print("Using device", device)

# Github URL where the dataset is stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial10/"
# Files to download
pretrained_files = [(DATASET_PATH, "TinyImageNet.zip"), (CHECKPOINT_PATH, "patches.zip")]
# Create checkpoint path if it doesn't exist yet
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# Mean and Std from ImageNet
NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])
# No resizing and center crop necessary as images are already preprocessed.
plain_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORM_MEAN,
                         std=NORM_STD)
])

# Load dataset and create data loader
imagenet_path = os.path.join(DATASET_PATH, "TinyImageNet/")
assert os.path.isdir(imagenet_path), f"Could not find the ImageNet dataset at expected path \"{imagenet_path}\". " + \
                                     f"Please make sure to have downloaded the ImageNet dataset here, or change the {DATASET_PATH=} variable."
dataset = torchvision.datasets.ImageFolder(root=imagenet_path, transform=plain_transforms)
data_loader = data.DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=8)

# Load CNN architecture pretrained on ImageNet
os.environ["TORCH_HOME"] = CHECKPOINT_PATH
pretrained_model = torchvision.models.resnet34(weights='IMAGENET1K_V1')
pretrained_model = pretrained_model.to(device)

# No gradients needed for the network
pretrained_model.eval()
for p in pretrained_model.parameters():
    p.requires_grad = False


# Load label names to interpret the label numbers 0 to 999
with open(os.path.join(imagenet_path, "label_list.json"), "r") as f:
    label_names = json.load(f)

def get_label_index(lab_str):
    assert lab_str in label_names, f"Label \"{lab_str}\" not found. Check the spelling of the class."
    return label_names.index(lab_str)

# For each file, check whether it already exists. If not, try downloading it.
for dir_name, file_name in pretrained_files:
    file_path = os.path.join(dir_name, file_name)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print("Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n", e)
        if file_name.endswith(".zip"):
            print("Unzipping file...")
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(file_path.rsplit("/",1)[0])

def eval_model(dataset_loader, img_func=None):
    tp, tp_5, counter = 0., 0., 0.
    for imgs, labels in tqdm(dataset_loader, desc="Validating..."):
        imgs = imgs.to(device)
        labels = labels.to(device)
        if img_func is not None:
            imgs = img_func(imgs, labels)
        with torch.no_grad():
            preds = pretrained_model(imgs)
        tp += (preds.argmax(dim=-1) == labels).sum()
        tp_5 += (preds.topk(5, dim=-1)[1] == labels[...,None]).any(dim=-1).sum()
        counter += preds.shape[0]
    acc = tp.float().item()/counter
    top5 = tp_5.float().item()/counter
    print(f"Top-1 error: {(100.0 * (1 - acc)):4.2f}%")
    print(f"Top-5 error: {(100.0 * (1 - top5)):4.2f}%")
    return acc, top5

def show_prediction(img, label, pred, K=5, adv_img=None, noise=None):
    if adv_img is not None:
        print("")
    #    save_image(adv_img, CHECKPOINT_PATH + "/" + f"adv_{label_names[label]} + .png")
     #   save_image(img, CHECKPOINT_PATH + "/" + f"{label_names[label]} + .png")
    if abs(pred.sum().item() - 1.0) > 1e-4:
        pred = torch.softmax(pred, dim=-1)
    topk_vals, topk_idx = pred.topk(K, dim=-1)
    topk_vals, topk_idx = topk_vals.cpu().numpy(), topk_idx.cpu().numpy()
    top_vals, top_idx = pred.topk(15, dim=-1)
    top_vals, top_idx = top_vals.cpu().numpy(), top_idx.cpu().numpy()
    print(f"{label_names[label]}")
    val = 0
    for i in range(15):
        if(label_names[top_idx[i]] == label_names[label]):
            val = top_vals[i] * 100

    print(f"{label_names[label]} | {val}")


    for i in range(K):
        print(f"{label_names[topk_idx[i]]} | {topk_vals[i] * 100}")
    print(val)
    if (label_names[topk_idx[i]] == label_names[label]):
        return [val, 1.0]
    else:
        return [val, 0.0]



#FGSM for normal attack
def fast_gradient_sign_method(model, imgs, labels, epsilon=0.02):
    # Determine prediction of the model
    inp_imgs = imgs.clone().requires_grad_()
    preds = model(inp_imgs.to(device))
    preds = F.log_softmax(preds, dim=-1)
    # Calculate loss by NLL
    loss = -torch.gather(preds, 1, labels.to(device).unsqueeze(dim=-1))
    loss.sum().backward()
    # Update image to adversarial example as written above
    noise_grad = torch.sign(inp_imgs.grad.to(imgs.device))
    fake_imgs = imgs + epsilon * noise_grad
    fake_imgs.detach_()
    return fake_imgs, noise_grad

#patch attack methods
def place_patch(img, patch):
    for i in range(img.shape[0]):
        h_offset = np.random.randint(0,img.shape[2]-patch.shape[1]-1)
        w_offset = np.random.randint(0,img.shape[3]-patch.shape[2]-1)
        img[i,:,h_offset:h_offset+patch.shape[1],w_offset:w_offset+patch.shape[2]] = patch_forward(patch)
    return img

TENSOR_MEANS, TENSOR_STD = torch.FloatTensor(NORM_MEAN)[:,None,None], torch.FloatTensor(NORM_STD)[:,None,None]
def patch_forward(patch):
    # Map patch values from [-infty,infty] to ImageNet min and max
    patch = (torch.tanh(patch) + 1 - 2 * TENSOR_MEANS) / (2 * TENSOR_STD)
    return patch

def eval_patch(model, patch, val_loader, target_class):
    model.eval()
    tp, tp_5, counter = 0., 0., 0.
    with torch.no_grad():
        for img, img_labels in tqdm(val_loader, desc="Validating...", leave=False):
            # For stability, place the patch at 4 random locations per image, and average the performance
            for _ in range(4):
                patch_img = place_patch(img, patch)
                patch_img = patch_img.to(device)
                img_labels = img_labels.to(device)
                pred = model(patch_img)
                # In the accuracy calculation, we need to exclude the images that are of our target class
                # as we would not "fool" the model into predicting those
                tp += torch.logical_and(pred.argmax(dim=-1) == target_class, img_labels != target_class).sum()
                tp_5 += torch.logical_and((pred.topk(5, dim=-1)[1] == target_class).any(dim=-1), img_labels != target_class).sum()
                counter += (img_labels != target_class).sum()
    acc = tp/counter
    top5 = tp_5/counter
    return acc, top5

def patch_attack(model, target_class, patch_size=64, num_epochs=5):
    # Leave a small set of images out to check generalization
    # In most of our experiments, the performance on the hold-out data points
    # was as good as on the training set. Overfitting was little possible due
    # to the small size of the patches.
    train_set, val_set = torch.utils.data.random_split(dataset, [4500, 500])
    train_loader = data.DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True, num_workers=8)
    val_loader = data.DataLoader(val_set, batch_size=32, shuffle=False, drop_last=False, num_workers=4)

    # Create parameter and optimizer
    if not isinstance(patch_size, tuple):
        patch_size = (patch_size, patch_size)
    patch = nn.Parameter(torch.zeros(3, patch_size[0], patch_size[1]), requires_grad=True)
    optimizer = torch.optim.SGD([patch], lr=1e-1, momentum=0.8)
    loss_module = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        t = tqdm(train_loader, leave=False)
        for img, _ in t:
            img = place_patch(img, patch)
            img = img.to(device)
            pred = model(img)
            labels = torch.zeros(img.shape[0], device=pred.device, dtype=torch.long).fill_(target_class)
            loss = loss_module(pred, labels)
            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()
            t.set_description(f"Epoch {epoch}, Loss: {loss.item():4.2f}")

    # Final validation
    acc, top5 = eval_patch(model, patch, val_loader, target_class)

    return patch.data, {"acc": acc.item(), "top5": top5.item()}

# Load evaluation results of the pretrained patches
json_results_file = os.path.join(CHECKPOINT_PATH, "patch_results.json")
json_results = {}
if os.path.isfile(json_results_file):
    with open(json_results_file, "r") as f:
        json_results = json.load(f)

# If you train new patches, you can save the results via calling this function
def save_results(patch_dict):
    result_dict = {cname: {psize: [t.item() if isinstance(t, torch.Tensor) else t
                                   for t in patch_dict[cname][psize]["results"]]
                           for psize in patch_dict[cname]}
                   for cname in patch_dict}
    with open(os.path.join(CHECKPOINT_PATH, "patch_results.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

def get_patches(class_names, patch_sizes):
    result_dict = dict()

    # Loop over all classes and patch sizes
    for name in class_names:
        result_dict[name] = dict()
        for patch_size in patch_sizes:
            c = label_names.index(name)
            file_name = os.path.join(CHECKPOINT_PATH, f"{name}_{patch_size}_patch.pt")
            # Load patch if pretrained file exists, otherwise start training
            if not os.path.isfile(file_name):
                patch, val_results = patch_attack(pretrained_model, target_class=c, patch_size=patch_size, num_epochs=5)
                print(f"Validation results for {name} and {patch_size}:", val_results)
                torch.save(patch, file_name)
            else:
                patch = torch.load(file_name)
            # Load evaluation results if exist, otherwise manually evaluate the patch
            if name in json_results:
                results = json_results[name][str(patch_size)]
            else:
                results = eval_patch(pretrained_model, patch, data_loader, target_class=c)

            # Store results and the patches in a dict for better access
            result_dict[name][patch_size] = {
                "results": results,
                "patch": patch
            }

    return result_dict

def perform_patch_attack(patch):
    patch_batch = exmp_batch.clone()
    patch_batch = place_patch(patch_batch, patch)
    return patch_batch

def perform_multi_patch_attack(patch):
    patch_batch = exmp_batch.clone()
    for i in range(4):
        patch_batch = place_patch(patch_batch, patch)
    return patch_batch

def gaussian_blur(img,  kernelSize, sig):
    blurrer = v2.GaussianBlur(kernel_size=kernelSize, sigma=sig)
    blurred_imgs = blurrer(img) 
    return img

def median_blur(input_tensor, kernel_size):
    pad_size = kernel_size // 2
    padded_input = F.pad(input_tensor, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
    unfolded = padded_input.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
    median = unfolded.contiguous().view(*unfolded.shape[:3], -1).median(dim=-1)[0]

    return median

def monte_denoise(image_tensor, beta):
    num_iterations=1000
    denoised_image = image_tensor.clone()

    # Get image dimensions
    C, H, W = denoised_image.shape

    # MCMC iterations
    for _ in range(num_iterations):
        # Randomly select a pixel
        i = torch.randint(0, H, (1,)).item()
        j = torch.randint(0, W, (1,)).item()

        i_min = max(i - 1, 0)
        i_max = min(i + 2, H)
        j_min = max(j - 1, 0)
        j_max = min(j + 2, W)
        neighbors = denoised_image[:, i_min:i_max, j_min:j_max]

        mean_neighborhood = neighbors.mean(dim=(1, 2))

        proposed_value = mean_neighborhood + beta * torch.randn(C)

        current_value = denoised_image[:, i, j]
        energy_diff = torch.norm(current_value - mean_neighborhood) - torch.norm(proposed_value - mean_neighborhood)

        if torch.exp(-energy_diff / beta) > torch.rand(1):
            denoised_image[:, i, j] = proposed_value

    return denoised_image

def defend_image_with_gan(img):
    gan_path =  "./APE-GAN/checkpoint/cifar/10.tar" #torch.load(args.gan_path)
    gan_point = torch.load(gan_path)

    G = Generator(3)
    G.load_state_dict(gan_point["generator"])
    loss_cre = nn.CrossEntropyLoss()

   # model.eval(), G.eval()
    G.eval()
    x_ape = G(img.unsqueeze(0))
    return x_ape

#running models
if __name__ ==  '__main__':
    #_ = eval_model(data_loader)
   
    file = open("output_data.csv",  "w")
    file.write("type,time,prediction,correct\n")
    exmp_batch, label_batch = next(iter(data_loader))
    print("Model with out attack")
    with torch.no_grad():
        preds = pretrained_model(exmp_batch.to(device))
    print(f"{exmp_batch=} + {label_batch=}")
    for i in range(1,17,5):
        output = show_prediction(exmp_batch[i], label_batch[i], preds[i])
        file.write(f"v,0,{output[0]},{output[1]}\n")
    print("Model with Traditional adv ttack")
    adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
    with torch.no_grad():
        adv_preds = pretrained_model(adv_imgs.to(device))

    for i in range(1,17,5):
        output = show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=adv_imgs[i], noise=noise_grad[i])
        file.write(f"tau,0,{output[0]},{output[1]}\n")

    print("Model with single patch attack")

    class_names = [ 'goldfish']
    patch_sizes = [32]

    patch_dict = get_patches(class_names, patch_sizes)
    patch_batch = perform_patch_attack(patch_dict['goldfish'][32]['patch'])
    with torch.no_grad():
        patch_preds = pretrained_model(patch_batch.to(device))
    for i in range(1,17,5):
        output = show_prediction(patch_batch[i], label_batch[i], patch_preds[i])
        file.write(f"spu,0,{output[0]},{output[1]}\n")
    
    print("Model with multi patch attack")
    multi_patch_batch = perform_multi_patch_attack(patch_dict['goldfish'][32]['patch'])

    with torch.no_grad():
        patch_preds = pretrained_model(multi_patch_batch.to(device))
    for i in range(1,17,5):
        output = show_prediction(multi_patch_batch[i], label_batch[i], patch_preds[i])
        file.write(f"mpu,0,{output[0]},{output[1]}\n")

    kernel_sizes = [3,5,7,15,31,61]
    sigmas = [1.0,2.0,3.0,5.0]
    betas = [.1,.2,.4,.6,.8]

    print("Median defended Model with adv attack")
    for kernel_size in kernel_sizes:    
        adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
        patch_batch = perform_patch_attack(patch_dict['goldfish'][32]['patch'])
        multi_patch_batch = perform_multi_patch_attack(patch_dict['goldfish'][32]['patch'])
        med_imgs = adv_imgs
        med_imgs0 = patch_batch 
        med_imgs1 = multi_patch_batch 
        timeTaken = 0
        start = time.time()
        for i in range(0, len(med_imgs)):
            med_imgs[i] = median_blur(med_imgs[i], kernel_size)
            med_imgs0[i] = median_blur(med_imgs0[i], kernel_size)
            med_imgs1[i] = median_blur(med_imgs1[i], kernel_size)
        timeTaken = (time.time() - start) / 3
        with torch.no_grad():
            adv_preds = pretrained_model(med_imgs.to(device))
            adv_preds0 = pretrained_model(med_imgs0.to(device))
            adv_preds1 = pretrained_model(med_imgs1.to(device))
        for i in range(1,17,5):
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=med_imgs[i], noise=noise_grad[i])
            file.write(f"tamed{kernel_size},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds0[i], adv_img=med_imgs0[i], noise=noise_grad[i])
            file.write(f"spmed{kernel_size},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds1[i], adv_img=med_imgs1[i], noise=noise_grad[i])
            file.write(f"mpmed{kernel_size},{timeTaken},{output[0]},{output[1]}\n")

    print("Gussian defended Model with adv attack")
    for rad in sigmas:    
        adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
        patch_batch = perform_patch_attack(patch_dict['goldfish'][32]['patch'])
        multi_patch_batch = perform_multi_patch_attack(patch_dict['goldfish'][32]['patch'])
        gus_imgs= adv_imgs
        gus_imgs0=patch_batch 
        gus_imgs1= multi_patch_batch
        timeTaken = 0
        start = time.time()
        for i in range(0, len(gus_imgs)):
            gus_imgs[i] = gaussian_blur(gus_imgs[i],5, rad)
            gus_imgs0[i] = gaussian_blur(gus_imgs0[i],5, rad)
            gus_imgs1[i] = gaussian_blur(gus_imgs1[i],5, rad)
        timeTaken = (time.time() - start) / 3
        with torch.no_grad():
            adv_preds = pretrained_model(gus_imgs.to(device))
            adv_preds0 = pretrained_model(gus_imgs0.to(device))
            adv_preds1 = pretrained_model(gus_imgs1.to(device))
        for i in range(1,17,5):
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=gus_imgs[i], noise=noise_grad[i])
            file.write(f"tagus{rad},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds0[i], adv_img=gus_imgs0[i], noise=noise_grad[i])
            file.write(f"spgus{rad},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds1[i], adv_img=gus_imgs1[i], noise=noise_grad[i])
            file.write(f"mpgus{rad},{timeTaken},{output[0]},{output[1]}\n")


    print("Mont defended Model with adv attack")
    for beta in betas:
        adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
        patch_batch = perform_patch_attack(patch_dict['goldfish'][32]['patch'])
        multi_patch_batch = perform_multi_patch_attack(patch_dict['goldfish'][32]['patch'])
        mont_imgs = adv_imgs
        mont_imgs0 = patch_batch
        mont_imgs1 = multi_patch_batch
        timeTaken = 0
        start = time.time()
        for i in range(0, len(mont_imgs)):
            mont_imgs[i] = monte_denoise(mont_imgs[i], beta)
            mont_imgs0[i] = monte_denoise(mont_imgs0[i], beta)
            mont_imgs1[i] = monte_denoise(mont_imgs1[i], beta)
        timeTaken = (time.time() - start) / 3
        with torch.no_grad():
            adv_preds = pretrained_model(mont_imgs.to(device))
            adv_preds0 = pretrained_model(mont_imgs0.to(device))
            adv_preds1 = pretrained_model(mont_imgs1.to(device))
        for i in range(1,17,5):
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=mont_imgs[i], noise=noise_grad[i])
            file.write(f"tamont{beta},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds0[i], adv_img=mont_imgs0[i], noise=noise_grad[i])
            file.write(f"spmont{beta},{timeTaken},{output[0]},{output[1]}\n")
            output = show_prediction(exmp_batch[i], label_batch[i], adv_preds1[i], adv_img=mont_imgs1[i], noise=noise_grad[i])
            file.write(f"mpmont{beta},{timeTaken},{output[0]},{output[1]}\n")

    print("GAN defended Model with adv attack")
    adv_imgs, noise_grad = fast_gradient_sign_method(pretrained_model, exmp_batch, label_batch, epsilon=0.02)
    patch_batch = perform_patch_attack(patch_dict['goldfish'][32]['patch'])
    multi_patch_batch = perform_multi_patch_attack(patch_dict['goldfish'][32]['patch'])
    gan_imgs= adv_imgs
    gan_imgs0=patch_batch 
    gan_imgs1= multi_patch_batch
    timeTaken = 0
    start = time.time()
    for i in range(0, len(gan_imgs)):
        gan_imgs[i] = defend_image_with_gan(gan_imgs[i])
        gan_imgs0[i] = defend_image_with_gan(gan_imgs0[i])
        gan_imgs1[i] = defend_image_with_gan(gan_imgs1[i])

    timeTaken = (time.time() - start) / 3
    with torch.no_grad():
        adv_preds = pretrained_model(gan_imgs.to(device))
        adv_preds0 = pretrained_model(gan_imgs0.to(device))
        adv_preds1 = pretrained_model(gan_imgs1.to(device))
    for i in range(1,17,5):
        output = show_prediction(exmp_batch[i], label_batch[i], adv_preds[i], adv_img=gan_imgs[i].detach(), noise=noise_grad[i])
        file.write(f"tagan,{timeTaken},{output[0]},{output[1]}\n")
        output = show_prediction(exmp_batch[i], label_batch[i], adv_preds0[i], adv_img=gan_imgs0[i].detach(), noise=noise_grad[i])
        file.write(f"spgan,{timeTaken},{output[0]},{output[1]}\n")
        output = show_prediction(exmp_batch[i], label_batch[i], adv_preds1[i], adv_img=gan_imgs1[i].detach(), noise=noise_grad[i])
        file.write(f"mpgan,{timeTaken},{output[0]},{output[1]}\n")
    file.close()
    print("DONE")
    time.sleep(600)
