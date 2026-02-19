
from .transforms import VAE_Transform, VAETransformMRI
from .local_path import get_data_path
import torch 
from torch.utils.data import Dataset
import random 
import time

class MultiModalDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        
        # print(f"get data from the dataloader....")
        # s = time.time()
        data = self.transform(self.data[index])
        # except:
        #     random_index = random.randint(0, len(self.data))
        #     data = self.transform(self.data[random_index])
        # e = time.time()
        # print(f"spend time data is {e - s}")

        return data


    def __len__(self) -> int:
        return len(self.data)

    
def get_transforms(args):
    train_transform = VAETransformMRI(
        is_train=True,
        random_aug=args.random_aug,  # whether apply random data augmentation for training
        k=4,  # patches should be divisible by k
        patch_size=args.patch_size,
        val_patch_size=args.val_patch_size,
        output_dtype=torch.float32,  # final data type
        spacing_type=args.spacing_type,
        spacing=args.spacing,
        image_keys=["t1c", "t1n", "t2f", "t2w"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    ) 

    val_transform = VAETransformMRI(
        is_train=False,
        random_aug=False,
        k=4,  # patches should be divisible by k
        val_patch_size=args.val_patch_size,  # if None, will validate on whole image volume
        output_dtype=torch.float16,  # final data type
        image_keys=["t1c", "t1n", "t2f", "t2w"],
        label_keys=[],
        additional_keys=[],
        select_channel=0,
    )
    return train_transform, val_transform

def get_dataset(args):
    train_files, val_files = get_data_path()
    
    train_transform, val_transform = get_transforms(args)

    print(f"Total number of training data is {len(train_files)}.")
    dataset_train = MultiModalDataset(data=train_files, transform=train_transform)
    dataset_val = MultiModalDataset(data=val_files, transform=val_transform)

    return dataset_train, dataset_val
