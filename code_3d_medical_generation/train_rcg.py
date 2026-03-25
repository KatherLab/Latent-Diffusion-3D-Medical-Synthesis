import numpy as np
import torch 
from light_training.trainer import Trainer
import os
from torch.cuda.amp import GradScaler, autocast
import os
import gc

import torch
from light_training.utils.files_helper import save_new_model_and_delete_last
from torch.optim import lr_scheduler
from monai.losses.perceptual import PerceptualLoss
from torch.cuda.amp import GradScaler, autocast
from rcg.rdm.util import instantiate_from_config
from omegaconf import OmegaConf
from rcg.rdm.models.diffusion.ddim import DDIMSampler
import os
from monai.inferers.inferer import SimpleInferer, SlidingWindowInferer
from scripts.utils import dynamic_infer
from torch.nn import L1Loss
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from utils.brain_data_utils import get_dataset, get_transforms
torch.multiprocessing.set_sharing_strategy('file_system')
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

args = OmegaConf.load("configs/config_unet3d_rcg.yaml")
device = "cuda:0"

train_dataset, val_dataset = get_dataset(args)
train_transform, val_transform = get_transforms(args)
# val_dataset = None 


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

# please adjust the learning rate warmup rule based on your dataset and n_epochs
def warmup_rule(epoch):
    # learning rate warmup rule
    if epoch < 10:
        return 0.01
    elif epoch < 20:
        return 0.1
    else:
        return 1.0

class PolyRule:
    def __init__(self, max_epochs, power=0.9):
        self.max_epochs = max_epochs
        self.power = power

    def __call__(self, epoch):
        return (1 - epoch / self.max_epochs) ** self.power
    

class MyTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/", master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port, training_script)
        self.best_metric = 1000
        self.scaler = GradScaler(init_scale=2.0**8, growth_factor=1.5)

        self.val_inferer = (
            SlidingWindowInferer(
                roi_size=args.val_patch_size,
                sw_batch_size=1,
                progress=True,
                overlap=0.0,
                device=torch.device("cpu"),
                sw_device=self.local_rank,
            )
        )
        
        from guided_diffusion.unet_rdm import UNetModel

        self.model = UNetModel(dims=3, image_size=96, in_channels=2,
                        model_channels=96, out_channels=2, 
                        num_res_blocks=1, attention_resolutions=[32,16,8],
                        channel_mult=[1, 2, 2, 2]).to(self.local_rank)

        config = OmegaConf.load("./config/rdm/rdm_model.yaml")
        rdm = instantiate_from_config(config.model)
        model_path = "/share/project/zhaohuxing/nvidia_generation/maisi/logs_four_datasets/unet3d_rdm_ep50/autoencoder.pt"
        rdm = load_state_dict(rdm, model_path)
        rdm.to(self.local_rank)
        rdm.eval()

        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr, eps=1e-06)
        self.intensity_loss = L1Loss(reduction="mean")

        self.loss_perceptual = (
            PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).eval().to(self.local_rank)
        )

        poly_rule_instance = PolyRule(max_epochs)
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=poly_rule_instance)

        self.sampler = DDIMSampler(model=rdm)

    def get_inputs(self, batch):
        image_t1n, image_t1c, image_t2w, image_t2f = batch["t1n"], batch["t1c"], batch["t2w"], batch["t2f"]
        inputs = torch.cat([image_t1n, image_t2w], dim=1)
        outputs = torch.cat([image_t1c, image_t2f], dim=1)
        return inputs, outputs
    
    def training_step(self, batch):
        images, labels = self.get_inputs(batch)

        self.optimizer.zero_grad()

        with autocast(enabled=True):    
            conditioning = images
            sampled_rep, _ = self.sampler.sample(10, conditioning=conditioning, batch_size=conditioning.shape[0],
                                                        shape=(192, 1, 1),
                                                        eta=1.0, verbose=False)
            sampled_rep = sampled_rep[:, :, 0, 0]                

            reconstruction = self.model(images, sampled_rep)

            loss_re = self.intensity_loss(reconstruction, labels)
            
            loss_p = self.loss_perceptual(reconstruction[:, 0:1].float(), labels[:, 0:1].float()) + self.loss_perceptual(reconstruction[:, 1:2].float(), labels[:, 1:2].float()) 
            
            loss = loss_re + 0.3 * loss_p

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        lr = self.optimizer.param_groups[0]["lr"]
        
        self.log("loss_re", loss_re, step=self.global_step)
        self.log("loss_p", loss_p, step=self.global_step)
        self.log("lr", lr, step=self.global_step)
            
        return loss 

    def epoch_end(self):
        # save model 
        if self.local_rank == 0:
            save_new_model_and_delete_last(self.model, 
                                        os.path.join(self.logdir, 
                                        f"final_model_.pt"), 
                                        delete_symbol="final_model")
        

        self.scheduler.step() 
    
    def validation_step(self, batch):

        with torch.no_grad():
            with autocast(enabled=True):            
                images = torch.cat([batch["t1n"], batch["t2w"]], dim=1)
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1)

                sampled_rep, _ = self.sampler.sample(10, conditioning=images, batch_size=images.shape[0],
                                                            shape=(192, 1, 1),
                                                            eta=1.0, verbose=False)

                sampled_rep = sampled_rep[:, :, 0, 0]              
                
                labels = labels.detach().cpu()
                reconstruction = dynamic_infer(self.val_inferer, self.model, images, rdm_rep=sampled_rep)
                mae = torch.nn.functional.l1_loss(reconstruction, labels, reduction='mean')
        
        print(f"mae is {mae}")
        torch.cuda.empty_cache()
        # only val single sample.
        self.break_validation = True

        return mae.item()                

    def validation_end(self, val_outputs):
        mae = val_outputs.mean()

        print(f"validation mae is {mae}") 
        
        self.log("mae", mae, step=self.epoch)

        if mae < self.best_metric:
            self.best_metric = mae
            save_new_model_and_delete_last(self.model, 
                                            os.path.join(self.logdir, 
                                            f"best_model_{mae:.4f}.pt"), 
                                            delete_symbol="best_model")

        save_new_model_and_delete_last(self.model, 
                                        os.path.join(self.logdir, 
                                        f"final_model_{mae:.4f}.pt"), 
                                        delete_symbol="final_model")
        

if __name__ == "__main__":
    trainer = MyTrainer(env_type=args.env,
                            max_epochs=args.max_epochs,
                            batch_size=args.batch_size,
                            device=device,
                            logdir=args.logdir,
                            val_every=args.val_every,
                            num_gpus=args.num_gpus,
                            master_port=17751,
                            training_script=__file__)
    
    trainer.train(train_dataset=train_dataset, val_dataset=None)