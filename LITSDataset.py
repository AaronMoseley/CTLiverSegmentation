import h5py
from torch.utils.data import Dataset
import torch

class LITSBinaryDataset(Dataset):
    def __init__(self, fileName):
        super().__init__()

        #Keeps a file pointer open throughout use
        self.file = h5py.File(fileName, 'r')

        #Precalculates length to reduce training computations
        self.length = len(list(self.file.keys()))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        data = self.file["Slice" + str(idx)]["Slice"]
        segmentation = self.file["Slice" + str(idx)]["Segmentation"]
        label = self.file["Slice" + str(idx)].attrs.get("ImageLabel")

        result = []

        #Returns list containing slice data and image label
        #Does not currently return segmentation data, will need to implement for decoder
        result.append(torch.Tensor(data[...]).unsqueeze(0))
        result.append(torch.clamp(torch.Tensor(segmentation[...]).unsqueeze(0), min=0, max=1))
        result.append(torch.Tensor(label).squeeze(0))

        return result

    def closeFile(self):
        #Closes file once dataset is no longer being used
        #Do not use class instance after this function is called
        self.file.close()