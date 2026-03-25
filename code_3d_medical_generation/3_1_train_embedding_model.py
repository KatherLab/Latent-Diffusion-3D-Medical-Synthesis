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
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
from utils.brain_data_utils import get_dataset
warnings.filterwarnings("ignore")
import torch.multiprocessing as mp
import torch.distributed as dist

from guided_diffusion.unet_raw_3d import UNetModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def cleanup():
    dist.destroy_process_group()

args = OmegaConf.load("configs/config_unet3d_rcg.yaml")


def patchify_3d(in_channels, imgs, patch_size):
    """
    imgs: (N, C, D, H, W)
    x: (N, L, patch_size_d * patch_size_h * patch_size_w * C)
    """
    pd, ph, pw = patch_size
    assert imgs.shape[2] % pd == 0 and imgs.shape[3] % ph == 0 and imgs.shape[4] % pw == 0
    d = imgs.shape[2] // pd
    h = imgs.shape[3] // ph
    w = imgs.shape[4] // pw
    x = imgs.reshape(imgs.shape[0], in_channels, d, pd, h, ph, w, pw)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    x = x.reshape(imgs.shape[0], d * h * w, pd * ph * pw * in_channels)
    return x


def unpatchify_3d(in_channels, x, patch_size, grid_size):
    """
    x: (N, L, patch_size_d * patch_size_h * patch_size_w * C)
    imgs: (N, C, D, H, W)
    """
    pd, ph, pw = patch_size
    d, h, w = grid_size
    assert d * h * w == x.shape[1]
    x = x.reshape(x.shape[0], d, h, w, pd, ph, pw, in_channels)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    imgs = x.reshape(x.shape[0], in_channels, d * pd, h * ph, w * pw)
    return imgs


def random_masking_3d(x, mask_ratio):
    """
    x: [N, L, D], sequence
    """
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))
    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return x_masked, mask, ids_restore


def mask_func_3d(x, in_channels, mask_ratio, patch_size, mask_value=0.0):
    """
    3D mask function for volumetric data.
    x: (N, C, D, H, W)
    patch_size: (pd, ph, pw)
    """
    batch = x.shape[0]
    pd, ph, pw = patch_size
    d = x.shape[2] // pd
    h = x.shape[3] // ph
    w = x.shape[4] // pw
    grid_size = (d, h, w)

    x_patch = patchify_3d(in_channels, x, patch_size)
    mask_patch, mask, ids_restore = random_masking_3d(x_patch, mask_ratio)
    mask_tokens = torch.ones(1, 1, in_channels * pd * ph * pw) * mask_value
    mask_tokens = mask_tokens.to(x.device)
    mask_tokens = mask_tokens.repeat(batch, ids_restore.shape[1] - mask_patch.shape[1], 1)

    x_ = torch.cat([mask_patch, mask_tokens], dim=1)
    x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, mask_patch.shape[2]))
    x = unpatchify_3d(in_channels, x_, patch_size=patch_size, grid_size=grid_size)
    return x, mask


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

    logdir = "./logs/embedding_model_3d"

    if local_rank == 0:
        Path(logdir).mkdir(parents=True, exist_ok=True)
        tensorboard_writer = SummaryWriter(logdir)
        print(f"Tensorboard event will be saved as {logdir}.")
    else:
        tensorboard_writer = None

    if local_rank == 0:
        trained_model_path = os.path.join(logdir, "embedding_model.pt")
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

    model = UNetModel(image_size=96, in_channels=2,
                    model_channels=96, out_channels=2,
                    num_res_blocks=1, attention_resolutions=[32, 16, 8],
                    dims=3, channel_mult=[1, 2, 2, 2]).to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                        device_ids=[local_rank],
                                                        output_device=local_rank,
                                                        find_unused_parameters=True)

    mse_loss = nn.MSELoss()

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

    # 3D patch size for masking: 96 / 24 = 4 patches per dim, 4*4*4 = 64 patches total
    mask_patch_size = (24, 24, 24)
    mask_ratio = 0.75

    for epoch in range(start_epoch, max_epochs):

        model.train()
        train_epoch_loss = 0

        with tqdm(total=len(dataloader_train), disable=(local_rank != 0)) as t:

            t.set_description('Epoch %i' % epoch)

            for batch in dataloader_train:

                # 50% chance to encode target modality, 50% source modality
                if torch.rand(1).item() < 0.5:
                    outputs = torch.cat([batch["t1c"], batch["t2f"]], dim=1).to(device)
                else:
                    outputs = torch.cat([batch["t1n"], batch["t2w"]], dim=1).to(device)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=True):
                    outputs_in, _ = mask_func_3d(outputs, 2, mask_ratio, mask_patch_size)

                    pred = model(outputs_in)

                    loss = mse_loss(pred, outputs)

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
