import os
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from PIL import Image


class MFI_Dataset(Dataset):
    def __init__(self, datasetPath, phase, use_dataTransform, resize, imgSzie):
        super(MFI_Dataset, self).__init__()
        self.datasetPath = datasetPath
        self.phase = phase
        self.use_dataTransform = use_dataTransform
        self.resize = resize
        self.imgSzie = imgSzie

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)])


    def __len__(self):
        # dirsName = os.listdir(os.path.join(self.datasetPath, "input/train"))
        # assert len(dirsName) >= 2, "Please check that the dataset is formatted correctly."
        if self.phase == "train":
            dirsPath = os.path.join(os.path.join(self.datasetPath, "gt/test"))
            return len(os.listdir(dirsPath))
        else:
            dirsPath = os.path.join(os.path.join(self.datasetPath, "input/train"))
            return len(os.listdir(dirsPath))

    def __getitem__(self, index):
        if self.phase == "train":
            clearImg_dirPath = os.path.join(self.datasetPath, "gt/test")
            # source image1
            sourceImg_dirPath = os.path.join(self.datasetPath, "input/test")
            sourceImg_names = os.listdir(clearImg_dirPath)
            sourceImg_path = os.path.join(sourceImg_dirPath, sourceImg_names[index])
            sourceImg = cv2.imread(sourceImg_path)

            # gt
            
            clearImg_names = os.listdir(clearImg_dirPath)
            clearImg_path = os.path.join(clearImg_dirPath, clearImg_names[index])
            clearImg = cv2.imread(clearImg_path)

            if self.resize:
                sourceImg = cv2.resize(sourceImg, (self.imgSzie, self.imgSzie))
                clearImg = cv2.resize(clearImg, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg = self.transform(sourceImg)
                clearImg = self.transform(clearImg)

            return [sourceImg, clearImg]

        elif self.phase == "valid":
            # source image1
            sourceImg_dirPath = os.path.join(self.datasetPath, "input/test")
            sourceImg_names = os.listdir(sourceImg_dirPath)
            sourceImg_path = os.path.join(sourceImg_dirPath, sourceImg_names[index])
            sourceImg = cv2.imread(sourceImg_path)


            if self.resize:
                sourceImg = cv2.resize(sourceImg, (self.imgSzie, self.imgSzie))
            if self.use_dataTransform:
                sourceImg = self.transform(sourceImg)

            return [sourceImg, sourceImg_names[index]]