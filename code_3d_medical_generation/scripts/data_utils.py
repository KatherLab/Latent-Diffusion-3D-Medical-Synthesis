
from torch.utils.data import Dataset
import random 

class MultiModalDataset(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):

        # try:
        data = self.transform(self.data[index])
        # except:
        #     random_index = random.randint(0, len(self.data))
        #     data = self.transform(self.data[random_index])
            
        return data


    def __len__(self) -> int:
        return len(self.data)


class MultiModalDataset2D(Dataset):
    def __init__(self, data, transform) -> None:
        super().__init__()
        self.data = data
        self.transform = transform

    def __getitem__(self, index):
        
        data = self.transform(self.data[index])

        t1n = data["t1n"] ## (1, 155, 240, 240)
        length = t1n.shape[1]


        random_int = random.randint(0, length)

        t1n = data["t1n"][:, random_int]
        t1c = data["t1c"][:, random_int]
        t2w = data["t2w"][:, random_int]
        t2f = data["t2f"][:, random_int]

        return [t1n, t1c, t2w, t2f]
    

    def __len__(self) -> int:
        return len(self.data)
    

