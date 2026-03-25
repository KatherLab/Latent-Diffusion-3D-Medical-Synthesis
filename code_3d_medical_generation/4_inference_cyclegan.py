
import argparse
import os
import torch
from omegaconf import OmegaConf
from guided_diffusion.unet_rdm import UNetModel
from monai.config import print_config
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from scripts.utils import dynamic_infer
import numpy as np 
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm 
import warnings
import SimpleITK as sitk 
from rcg.rdm.util import instantiate_from_config
from omegaconf import OmegaConf
from rcg.rdm.models.diffusion.ddim import DDIMSampler
from utils.brain_data_utils import get_transforms

import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

warnings.filterwarnings("ignore")

config_file = "./configs/config_cyclegan.yaml"

args = OmegaConf.load(config_file)

train_transform, val_transform = get_transforms(args)

device = "cuda:0"

from models.cyclegan3d import Generator3D
model = Generator3D(input_nc=2, output_nc=2).to(device)

def load_state_dict(model, weight_path, strict=True):
    sd = torch.load(weight_path, map_location="cpu")
    if "module" in sd :
        sd = sd["module"]
    new_sd = {}
    for k, v in sd.items():
        k = str(k)
        new_k = k[7:] if k.startswith("module") else k 
        new_sd[new_k] = v 

    model.load_state_dict(new_sd, strict=strict)
    print(f"loading parameters for rdm model...")
    return model 


model = load_state_dict(model, "/share/project/zhaohuxing/nvidia_generation/proj_maisi/logs/cyclegan3d_ep200/autoencoder.pt")
model.eval()

def save_to_nii(return_output,
                save_path,
                raw_spacing=[1,1,1],
                ):
    # return_output = return_output.astype(np.uint8)
    print(f" get image from array ...", return_output.shape)
    return_output = sitk.GetImageFromArray(return_output)
    if isinstance(raw_spacing[0], torch.Tensor):
        raw_spacing = [raw_spacing[0].item(), raw_spacing[1].item(), raw_spacing[2].item()]

    return_output.SetSpacing((raw_spacing[0], raw_spacing[1], raw_spacing[2]))

    sitk.WriteImage(return_output, save_path)

    print(f"{save_path} is saved successfully")

print(f"val patch size is {args.val_patch_size}")
val_inferer = SlidingWindowInferer(
        roi_size=args.val_patch_size,
        sw_batch_size=1,
        progress=True,
        overlap=0.6,
        device=torch.device("cpu"),
        sw_device=device,
    )


def inference(val_data, save_dir):
    for batch in tqdm(val_data, total=len(val_data)):

        path = batch["t1c"]
        case_name = path.split("/")[-2]

        save_last_dir = os.path.join(save_dir, case_name)
        os.makedirs(save_last_dir, exist_ok=True)

        batch = val_transform(batch)

        with torch.no_grad():
            with autocast(enabled=True):

                images = torch.cat([batch["t1n"], batch["t2w"]], dim=0).to(device)[None, ]
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=0)[None, ]
    
                reconstruction = val_inferer(images, model)

                print(reconstruction.shape)

                save_to_nii(reconstruction[0, 0].cpu().numpy(), os.path.join(save_last_dir, "t1c_gen.nii.gz"))
                save_to_nii(reconstruction[0, 1].cpu().numpy(), os.path.join(save_last_dir, "t2f_gen.nii.gz"))

                save_to_nii(labels[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1c.nii.gz"))
                save_to_nii(labels[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2f.nii.gz"))

                save_to_nii(images[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1n.nii.gz"))
                save_to_nii(images[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2w.nii.gz"))
            

def inference_egd(val_data, save_dir):
    for batch in tqdm(val_data, total=len(val_data)):

        path = batch["t1c"]
        case_name = path.split("/")[-4]

        save_last_dir = os.path.join(save_dir, case_name)
        os.makedirs(save_last_dir, exist_ok=True)

        print(f"paths is {batch}")

        raw = sitk.ReadImage(path)
        # Optionally convert to NumPy array
        image_array = sitk.GetArrayFromImage(raw)  # shape: [slices, height, width]

        print(image_array.shape)

        batch = val_transform(batch)

        t1n = batch["t1n"]

        print(t1n.shape)

        exit(0)

        with torch.no_grad():
            with autocast(enabled=True):
                
                if os.path.exists(os.path.join(save_last_dir, "t2w.nii.gz")):
                    continue
                
                images = torch.cat([batch["t1n"], batch["t2w"]], dim=0).to(device)[None, ]
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=0).to(device)[None, ]

                conditioning = images
                sampled_rep, _ = sampler.sample(10, conditioning=conditioning, batch_size=conditioning.shape[0],
                                                            shape=(192, 1, 1),
                                                            eta=1.0, verbose=False)

                sampled_rep = sampled_rep[:, :, 0, 0]                
                
                reconstruction = dynamic_infer(val_inferer, model, images, rdm_rep=sampled_rep)

                print(reconstruction.shape)

                save_to_nii(reconstruction[0, 0].cpu().numpy(), os.path.join(save_last_dir, "t1c_gen.nii.gz"))
                save_to_nii(reconstruction[0, 1].cpu().numpy(), os.path.join(save_last_dir, "t2f_gen.nii.gz"))

                save_to_nii(labels[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1c.nii.gz"))
                save_to_nii(labels[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2f.nii.gz"))

                save_to_nii(images[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1n.nii.gz"))
                save_to_nii(images[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2w.nii.gz"))


def inference_inhouse(val_data, save_dir):
    for batch in tqdm(val_data, total=len(val_data)):

        path = batch["t1c"]
        case_name = path.split("/")[-2]

        save_last_dir = os.path.join(save_dir, case_name)
        os.makedirs(save_last_dir, exist_ok=True)

        batch = val_transform(batch)

        with torch.no_grad():
            with autocast(enabled=True):
                
                if os.path.exists(os.path.join(save_last_dir, "t2w.nii.gz")):
                    continue
                
                images = torch.cat([batch["t1n"], batch["t2w"]], dim=0).to(device)[None, ]
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=0).to(device)[None, ]

                conditioning = images
                sampled_rep, _ = sampler.sample(10, conditioning=conditioning, batch_size=conditioning.shape[0],
                                                            shape=(192, 1, 1),
                                                            eta=1.0, verbose=False)

                sampled_rep = sampled_rep[:, :, 0, 0]                
                
                reconstruction = dynamic_infer(val_inferer, model, images, rdm_rep=sampled_rep)

                print(reconstruction.shape)

                save_to_nii(reconstruction[0, 0].cpu().numpy(), os.path.join(save_last_dir, "t1c_gen.nii.gz"))
                save_to_nii(reconstruction[0, 1].cpu().numpy(), os.path.join(save_last_dir, "t2f_gen.nii.gz"))

                save_to_nii(labels[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1c.nii.gz"))
                save_to_nii(labels[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2f.nii.gz"))

                save_to_nii(images[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1n.nii.gz"))
                save_to_nii(images[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2w.nii.gz"))

def inference_ucsf(val_data, save_dir):
    for batch in tqdm(val_data, total=len(val_data)):

        path = batch["t1c"]
        case_name = path.split("/")[-2]

        save_last_dir = os.path.join(save_dir, case_name)
        os.makedirs(save_last_dir, exist_ok=True)

        batch = val_transform(batch)

        with torch.no_grad():
            with autocast(enabled=True):
                
                if os.path.exists(os.path.join(save_last_dir, "t2w.nii.gz")):
                    continue
                
                images = torch.cat([batch["t1n"], batch["t2w"]], dim=0).to(device)[None, ]
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=0).to(device)[None, ]

                conditioning = images
                sampled_rep, _ = sampler.sample(10, conditioning=conditioning, batch_size=conditioning.shape[0],
                                                            shape=(192, 1, 1),
                                                            eta=1.0, verbose=False)

                sampled_rep = sampled_rep[:, :, 0, 0]                
                
                reconstruction = dynamic_infer(val_inferer, model, images, rdm_rep=sampled_rep)

                print(reconstruction.shape)

                save_to_nii(reconstruction[0, 0].cpu().numpy(), os.path.join(save_last_dir, "t1c_gen.nii.gz"))
                save_to_nii(reconstruction[0, 1].cpu().numpy(), os.path.join(save_last_dir, "t2f_gen.nii.gz"))

                save_to_nii(labels[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1c.nii.gz"))
                save_to_nii(labels[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2f.nii.gz"))

                save_to_nii(images[0, 0].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t1n.nii.gz"))
                save_to_nii(images[0, 1].cpu().numpy().astype(np.float32), os.path.join(save_last_dir, "t2w.nii.gz"))


from scripts.local_path import get_brats21_data, get_brats24_data, get_egd_data, get_gbm_data_path, get_ucsf_data_path, get_inhouse_data_path



_, val_files_brats21 = get_brats21_data()
inference(val_files_brats21, "predictions_brats21/cyclegan")

# _, val_files_brats24 = get_brats24_data()
# train_files_egd, val_files_egd = get_egd_data()
# _, val_files_gbm = get_gbm_data_path()
# train_ucsf_files, val_ucsf_files = get_ucsf_data_path()
# train_inhouse_files, val_inhouse_files = get_inhouse_data_path()

# print(train_inhouse_files, val_inhouse_files)
# inference_inhouse(train_inhouse_files, "predictions/inhouse_rcg")
# inference_inhouse(val_inhouse_files, "predictions/inhouse_rcg")


# inference(val_files_brats24, "predictions/brats24_rcg")
# inference_egd(val_files_egd, "predictions/egd_rcg")
# inference_egd(train_files_egd, "predictions/egd_rcg")

# inference(val_files_gbm, "predictions/gbm_rcg")

# inference_ucsf(train_ucsf_files, "predictions/ucsf_rcg")
# inference_ucsf(val_ucsf_files, "predictions/ucsf_rcg")




