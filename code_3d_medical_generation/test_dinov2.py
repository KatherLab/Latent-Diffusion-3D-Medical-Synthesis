import numpy as np
import torch 
import torch.nn as nn 
from utils.data_utils_brats23_no_norm import get_train_val_dataset
from light_training.trainer import Trainer
from utils.metrics import compute_psnr, compute_ssim
import random
import os 
from PIL import Image


data_dir = "./data/fullres/train"
fold = 0

env = "pytorch"
device = "cuda:0"
image_size = 518

class MyTrainer(Trainer):
    def __init__(self, env_type, max_epochs=1, batch_size=1, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        from model.semseg.dpt import DPT
        model_configs = {
                'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
        self.model = DPT(**{**model_configs['small'], 'nclass': 2})
        self.load_state_dict("/data/xingzhaohu/image_synthesis_miccai25/logs/dinov2_small/model/final_model_32.8851.pt")


    def get_inputs(self, batch):
        image_t1n, image_t1c, image_t2w, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        inputs = torch.cat([image_t1n, image_t2w, image_t1n], dim=1)
        outputs = torch.cat([image_t1c, image_t2f], dim=1)
        return inputs, outputs
    
    def save_image(self, pred, save_path):
        pred = Image.fromarray(pred).convert("L")
        pred.save(save_path)

    def cal_metric(self, pred, gt):
        pred = pred.clamp(0, 1)

        pred = pred.cpu().numpy()[0]
        gt = gt.cpu().numpy()[0]
    
        psnr = compute_psnr(pred, gt)
        ssim = compute_ssim(pred, gt)
        mae = np.mean(np.abs(pred - gt))
        return psnr, ssim, mae  
    
    def validation_step(self, batch):
        index = batch["index"][0]

        index = int(index)
        inputs, outputs = self.get_inputs(batch)

        pred = self.model(inputs)
        image_t1c = outputs[:, 0:1]
        image_t2f = outputs[:, 1:2]
        
        pred_t1c = pred[:, 0:1]
        pred_t1c = pred_t1c.clamp(0, 1) * 255
        pred_t1c = pred_t1c.cpu().numpy()[0, 0]

        pred_t2f = pred[:, 1:2]
        pred_t2f = pred_t2f.clamp(0, 1) * 255
        pred_t2f = pred_t2f.cpu().numpy()[0, 0]
        
        index = batch["index"][0]
        identifier = batch["identifier"][0]
        save_dir = f"./predictions/dinov2/{identifier}/"
        os.makedirs(save_dir, exist_ok=True)

        print(f"index is {index}, identifier is {identifier}")

        self.save_image(pred_t1c, save_path=os.path.join(save_dir, f"{index}_t1c.png"))
        self.save_image(pred_t2f, save_path=os.path.join(save_dir, f"{index}_t2f.png"))

        image_t1c = image_t1c.clamp(0, 1) * 255

        image_t2f = image_t2f.clamp(0, 1) * 255
        image_t1c = image_t1c.cpu().numpy()[0, 0]
        image_t2f = image_t2f.cpu().numpy()[0, 0]

        self.save_image(image_t1c, save_path=os.path.join(save_dir, f"{index}_t1c_gt.png"))
        self.save_image(image_t2f, save_path=os.path.join(save_dir, f"{index}_t2f_gt.png"))

        return 1.0

if __name__ == "__main__":

    trainer = MyTrainer(env_type=env,
                        device=device,
                        master_port=17752,
                        training_script=__file__)
    
    train_ds, val_ds, seg_ds = get_train_val_dataset(image_size=image_size)

    v_mean, v_out = trainer.validation_single_gpu(seg_ds)

    print(v_mean)
