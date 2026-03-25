import os
from pathlib import Path
import torch
import torch.nn as nn
from monai.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
from omegaconf import OmegaConf
from rcg.rdm.util import instantiate_from_config
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from utils.brain_data_utils import get_dataset
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp
import torch.distributed as dist

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def cleanup():
    dist.destroy_process_group()

args = OmegaConf.load("configs/config_unet3d_rcg.yaml")


def train(local_rank, world_size):

    if world_size > 1:
        distributed = True
    else:
        distributed = False

    if distributed:
        torch.distributed.init_process_group(
            backend='nccl', rank=local_rank, world_size=world_size)

    device = torch.device(local_rank)
    print(f"local_rank is {local_rank}")

    dataset_train, dataset_val = get_dataset(args)

    logdir = "./logs/rdm_model_3d"

    if local_rank == 0:
        Path(logdir).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(logdir)
        print(f"Tensorboard event will be saved as {logdir}.")
    else:
        tensorboard_writer = None

    if local_rank == 0:
        trained_model_path = os.path.join(logdir, "rdm_model.pt")
        print(f"Trained model will be saved as {trained_model_path}.")

    if distributed:
        train_sampler = DistributedSampler(dataset=dataset_train, even_divisible=True, shuffle=True)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                    num_workers=4, sampler=train_sampler,
                                    drop_last=True)
    else:
        train_sampler = None
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                        num_workers=4, shuffle=True, drop_last=True)

    config = OmegaConf.load("./config/rdm/rdm_model.yaml")
    model = instantiate_from_config(config.model).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, eps=1e-06)

    class PolyRule:
        def __init__(self, max_epochs, power=0.9):
            self.max_epochs = max_epochs
            self.power = power

        def __call__(self, epoch):
            return (1 - epoch / self.max_epochs) ** self.power

    scaler = GradScaler(init_scale=2.0**8, growth_factor=1.5)

    total_step = 0
    start_epoch = 0
    max_epochs = args.n_epochs

    poly_rule_instance = PolyRule(max_epochs)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=poly_rule_instance)

    for epoch in range(start_epoch, max_epochs):

        model.train()
        train_epoch_loss = 0

        with tqdm(total=len(dataloader_train), disable=(local_rank != 0)) as t:

            t.set_description('Epoch %i' % epoch)

            for batch in dataloader_train:

                source_images = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device)
                target_images = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    rdm_batch = {
                        "image": target_images,
                        "source_image": source_images,
                    }
                    loss, _ = model(x=None, c=None, batch=rdm_batch)

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()

                total_step += 1

                if local_rank == 0:
                    train_epoch_loss += loss.item()
                    t.set_postfix(loss=loss.item(), lr=scheduler.get_lr())
                    t.update(1)

            scheduler.step()

            if local_rank == 0:
                train_epoch_loss /= len(dataloader_train)
                print(f"Epoch {epoch} train_loss: {train_epoch_loss}.")
                tensorboard_writer.add_scalar("train_epoch_loss", train_epoch_loss, epoch)
                torch.save(model.state_dict(), trained_model_path)
                print("Save trained model to", trained_model_path)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"world size is {world_size}")

    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)
