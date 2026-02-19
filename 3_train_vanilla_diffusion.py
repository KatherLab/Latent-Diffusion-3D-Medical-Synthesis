import os
from pathlib import Path
import torch
from monai.data import DataLoader, DistributedSampler
from monai.losses.perceptual import PerceptualLoss
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm 
import warnings
from rcg.rdm.util import instantiate_from_config
from omegaconf import OmegaConf
from rcg.rdm.models.diffusion.ddim import DDIMSampler
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from utils.brain_data_utils import get_dataset, get_transforms
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp
import torch.distributed as dist


def cleanup():
    dist.destroy_process_group()

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
    # print(f"loading parameters for rdm model...")
    return model

args = OmegaConf.load("configs/config_unet3d_rcg.yaml")

def train(local_rank, world_size):

    if world_size > 1:
        distributed = True
    else :
        distributed = False

    if distributed:
        torch.distributed.init_process_group(
            backend='nccl', rank=local_rank, world_size=world_size) 
        # print(f"world size is {world_size}")

    device = torch.device(local_rank)

    print(f"local_rank is {local_rank}")

    dataset_train, dataset_val = get_dataset(args)

    if local_rank == 0:
        # initialize tensorboard writer
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(args.logdir)
        print(f"Tensorboard event will be saved as {args.logdir}.")
    else:
        tensorboard_writer = None

    # model path
    if local_rank == 0:
        trained_g_path = os.path.join(args.logdir, "autoencoder.pt")
        print(f"Trained model will be saved as {trained_g_path}.")

    if distributed:
        train_sampler = DistributedSampler(dataset=dataset_train, even_divisible=True, shuffle=True)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                                    num_workers=8, sampler=train_sampler, 
                                    drop_last=True)
    else:
        train_sampler = None
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                                        num_workers=8, shuffle=True, drop_last=True)
        

    from models.vanilla_diffusion.model import DiffUNet
    model = DiffUNet().to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True) 
            
    intensity_loss = L1Loss(reduction="mean")

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1e-06)

    scaler = GradScaler(init_scale=2.0**8, growth_factor=1.5)

    best_val_recon_epoch_loss = 10000000.0
    total_step = 0
    start_epoch = 0
    max_epochs = args.n_epochs
    # Training and validation loops
    for epoch in range(start_epoch, max_epochs):
        
        model.train()
        train_epoch_loss = 0

        with tqdm(total=len(dataloader_train), disable=(local_rank != 0)) as t:
            
            t.set_description('Epoch %i' % epoch)

            for batch in dataloader_train:
                
                images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device)
                labels = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device)
            
                optimizer.zero_grad(set_to_none=True)
                
                with autocast(enabled=True):
                    # Train Generator
                    x_t, time, noise = model(labels, pred_type="q_sample")
        
                    pred_xstart = model(x=x_t, step=time, image=images, pred_type="denoise")

                    loss = intensity_loss(pred_xstart.float(), labels.float())
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()

                total_step += 1
                
                if local_rank == 0:
                    train_epoch_loss += loss.item()
                    t.set_postfix(loss=loss.item())
                    t.update(1)
                
            if local_rank == 0:
                train_epoch_loss /= len(dataloader_train)
                print(f"Epoch {epoch} train_vae_loss: {train_epoch_loss}.")
                tensorboard_writer.add_scalar(f"train_epoch_loss", train_epoch_loss, epoch)
                torch.save(model.state_dict(), trained_g_path)
                print("Save trained autoencoder to", trained_g_path)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"world size is {world_size}")

    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)