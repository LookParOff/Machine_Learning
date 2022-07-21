# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from pycocotools.coco import COCO  # API for interact with dataset

from PIL import Image  # to load image

# to show image
import matplotlib.pyplot as plt
import skimage.io as io
# neural networks
import torch
from torch.utils.data import IterableDataset, DataLoader
import torchvision
from torchmetrics import JaccardIndex
from tqdm import tqdm


def get_person_annotations(coco):
    person_id = coco.getCatIds(catNms=["person"])[0]
    ids_images_with_persons = coco.getImgIds(catIds=[person_id])
    f"{len(ids_images_with_persons)} images with person on it"

    annotations_ids = []
    for id_img in ids_images_with_persons:
        anns = coco.getAnnIds(imgIds=[id_img])
        annotations_ids.append(anns)
    f"{len(annotations_ids)} images with person on it"
    return annotations_ids, person_id


def get_filtered_data(coco):
    annotations_ids, person_id = get_person_annotations(coco)
    filtered_ann = []
    for num_iter, ann_id in enumerate(annotations_ids):
        skip_image = False
        count_of_persons_on_image = 0
        annotations_of_img = coco.loadAnns(ann_id)
        if len(annotations_of_img) > 15:
            # too many objects
            continue
        img = coco.loadImgs(annotations_of_img[0]["image_id"])[0]
        img_area = int(img["height"]) * int(img["width"])
        area_of_person = -1
        max_area = -1
        index_person_in_ann = 0
        for i, ann in enumerate(annotations_of_img):
            if ann["category_id"] == person_id:
                count_of_persons_on_image += 1
                area_of_person = ann["area"]
                index_person_in_ann = i
            if max_area < ann["area"]:
                max_area = ann["area"]
            if count_of_persons_on_image > 1:
                skip_image = True
                break

        if skip_image:
            continue
        if area_of_person < img_area * 0.10:
            continue
        if area_of_person != max_area:
            continue
        if num_iter % 1000:
            print(f"iter is {num_iter}", end="\r")
        filtered_ann.append(annotations_of_img[index_person_in_ann])
    #     I = io.imread(img['coco_url'])
    #     plt.imshow(I)
    #     coco.showAnns(annotations_of_img)
    #     plt.show()
    f"{len(filtered_ann)} filtered images with person on it"
    return filtered_ann


def show_filtered_images(coco, filtered_ann):
    row_n = 4
    col_n = 7
    fig = plt.figure(figsize=(30, 8))
    for index in range(1, row_n * col_n + 1):
        annotation_of_img = filtered_ann[np.random.randint(0, len(filtered_ann))]
        img = coco.loadImgs(annotation_of_img["image_id"])[0]
        I = io.imread(img['coco_url'])
        plt.subplot(row_n, col_n, index)
        plt.imshow(I)
        plt.axis("off")
        img_area = int(img["height"]) * int(img["width"])
        person_area = annotation_of_img["area"]
        plt.title(f"persent of person {round(person_area / img_area, 3)}")
    plt.show()


def EDA_of_images_sizes(coco, filtered_ann):
    heights, widths = [], []
    for ann in filtered_ann:
        img = coco.loadImgs(ann["image_id"])[0]
        h, w = int(img["height"]), int(img["width"])
        heights.append(h)
        widths.append(w)

    unique_h, counts_h = np.unique(heights, return_counts=True)
    unique_w, counts_w = np.unique(widths, return_counts=True)

    fig = plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.scatter(heights, widths)
    plt.xlabel("height")
    plt.ylabel("widths")

    plt.subplot(2, 2, 2)
    plt.plot(unique_h, counts_h)
    plt.xlabel("height")
    plt.ylabel("count")

    plt.subplot(2, 2, 3)
    plt.plot(unique_w, counts_w)
    plt.xlabel("width")
    plt.ylabel("count")

    plt.show()


def check_of_loader_work(coco, filtered_ann, device):
    dataset = IterableDatasetCOCO(coco, filtered_ann, (300, 300), device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=3)
    plt.figure(figsize=(20, 5))
    for i, data in enumerate(dataloader):
        if i == 3:
            break
        x, y = data
        plt.subplot(1, 6, i * 2 + 1)
        plt.imshow(x[0][0])
        plt.subplot(1, 6, i * 2 + 2)
        plt.imshow(y[0][0])
    plt.show()
    print()


class IterableDatasetCOCO(IterableDataset):
    def __init__(self, coco, annotations, resize_shape, device):
        super().__init__()
        self.annotation_of_images = annotations
        self.resize_shape = resize_shape
        self.device = device
        self.coco = coco
        self.names_of_images = []
        self.__index = 0

        self.to_tensor = torchvision.transforms.ToTensor()
        self.resize = torchvision.transforms.Resize(self.resize_shape)

        for ann in self.annotation_of_images:
            # recording names of files, which will go to the network
            img_id = ann["image_id"]
            img = coco.loadImgs([img_id])
            self.names_of_images.append(img[0]["file_name"])

    def get_next_data(self):
        worker_info = torch.utils.data.get_worker_info()
        start = 0
        end = len(self.names_of_images)
        if worker_info is not None:
            per_worker = int(np.ceil((end - start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            start = worker_id * per_worker
            end = min(start + per_worker, end)
        names = self.names_of_images[start:end]
        for i, name in enumerate(names):
            # tensor_image = torch.empty((3, self.resize_shape[0], self.resize_shape[1]),
            #                            device=self.device)  # (C, H, W)
            # tensor_mask = torch.empty((1, self.resize_shape[0], self.resize_shape[1]),
            #                           device=self.device)
            name = self.names_of_images[start + i]
            img = Image.open(f"Datasets/COCO persons/{name}")
            # tensor_image = self.to_tensor(img)
            tensor_image = torch.tensor(np.array(img), device=self.device, dtype=torch.float32)
            if len(tensor_image.shape) != 3:
                # skip black and white  image
                continue
            tensor_image = tensor_image.permute((2, 0, 1))

            # mean = torch.mean(tensor_image, dim=[1, 2])
            # std = torch.std(tensor_image, dim=[1, 2])
            # calculated empirical by training dataset
            mean = [113.9068, 90.2008, 74.9105]
            std = [51.5830, 61.2258, 64.5600]
            norm = torchvision.transforms.Normalize(mean, std)
            tensor_image = norm(tensor_image)
            tensor_image = self.resize(tensor_image)

            image_mask = self.coco.annToMask(self.annotation_of_images[start + i])
            tensor_mask_person = torch.tensor(image_mask, device=self.device, dtype=torch.float32)
            tensor_mask_person = torch.unsqueeze(tensor_mask_person, dim=0)
            tensor_mask_person = self.resize(tensor_mask_person)
            tensor_mask_background = torch.abs(tensor_mask_person - 1)
            tensor_mask = torch.concat([tensor_mask_person, tensor_mask_background], dim=0)
            yield tensor_image, tensor_mask

    def __iter__(self):
        return iter(self.get_next_data())


class Block(torch.nn.Module):
    def __init__(self, inp_size, out_size, device):
        super().__init__()
        # with padding=1 output shape of result
        # will be the same with input
        self.conv1 = torch.nn.Conv2d(inp_size, out_size,
                                     kernel_size=(3, 3), padding=1, device=device)
        self.conv2 = torch.nn.Conv2d(out_size, out_size,
                                     kernel_size=(3, 3), padding=1, device=device)
        self.ReLU = torch.nn.ReLU()

    def forward(self, x):
        out = self.ReLU(self.conv1(x))
        out = self.ReLU(self.conv2(out))
        return out


class Encoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.max_pooling = torch.nn.MaxPool2d((2, 2))
        self.block_sizes = [3, 64, 128]
        self.blocks = [Block(self.block_sizes[i], self.block_sizes[i + 1], device)
                       for i, _ in enumerate(self.block_sizes[:-1])]

    def forward(self, x):
        copy_crops = []  # for operation copy and crop
        out = x
        for block in self.blocks[:-1]:
            out = block(out)
            copy_crops.append(out)
            out = self.max_pooling(out)

        out = self.blocks[-1](out)
        return out, copy_crops


class Decoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.block_sizes = [128, 64]
        self.blocks = [Block(self.block_sizes[i], self.block_sizes[i + 1], device)
                       for i, _ in enumerate(self.block_sizes[:-1])]
        self.transpose_convs = [torch.nn.ConvTranspose2d
                                (self.block_sizes[i], self.block_sizes[i + 1],
                                 kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), device=device)
                                for i, _ in enumerate(self.block_sizes[:-1])]

    def forward(self, x, copy_crops):
        # copy_crops for skip connection
        out = x
        for block, transpose_conv, skip in zip(self.blocks, self.transpose_convs, copy_crops):
            # print("out shape", out.shape)
            out = transpose_conv(out)
            # print("out shape trans", out.shape)
            out_width, out_height = out.shape[2:]
            skip_width, skip_height = skip.shape[2:]
            croped_height = (
                (skip_height - out_height) // 2, out_height + (skip_height - out_height) // 2)
            croped_width = (
                (skip_width - out_width) // 2, out_width + (skip_width - out_width) // 2)
            skip = skip[:, :, croped_height[0]:croped_height[1], croped_width[0]:croped_width[1]]
            # print("skip croped shape", skip.shape)
            out = torch.concat((skip, out), dim=1)
            # print("out after concat", out.shape)
            out = block(out)
            # print()
        return out


class UNet(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.encoder = Encoder(device)
        self.decoder = Decoder(device)
        self.conv1 = torch.nn.Conv2d(64, 2, kernel_size=(1, 1), device=device)
        self.SoftMax = torch.nn.Softmax(dim=1)  # along chanel dimension

    def forward(self, x):
        out, copy_crops = self.encoder(x)
        copy_crops = copy_crops[::-1]

        out = self.decoder(out, copy_crops)
        out = self.conv1(out)
        out = self.SoftMax(out)
        return out


def main(coco_module):
    device = torch.device("cuda")
    filtered_annotation = get_filtered_data(coco_module)
    # filtered_annotation = filtered_annotation[:100]
    length_train_data = int(len(filtered_annotation) * 0.8)
    length_test_data = len(filtered_annotation) - length_train_data
    dataset_train = IterableDatasetCOCO(coco_module, filtered_annotation[:length_train_data],
                                        (300, 300), device)
    dataset_test = IterableDatasetCOCO(coco_module, filtered_annotation[length_train_data:],
                                       (300, 300), device)
    batch_size = 10
    train_dataloader = DataLoader(dataset_train, batch_size=batch_size, num_workers=0)
    test_dataloader = DataLoader(dataset_test, batch_size=1, num_workers=0)
    print(train_dataloader.batch_size)
    unet = UNet(device)
    loss_func = torch.nn.CrossEntropyLoss()

    optm = torch.optim.Adam(unet.parameters(), lr=0.0001)
    metric_jaccard_macro = JaccardIndex(2, average="macro").to(device=device)
    metric_jaccard_micro = JaccardIndex(2, average="micro").to(device=device)

    history_loss = []
    history_jac_mac = []
    history_jac_mic = []

    epochs = 10
    for ep in range(epochs):
        for x, y in tqdm(train_dataloader, leave=True, total=length_train_data // batch_size):
            # скорее всего нужно убрать размерность для канала в y
            # или сделать их две) тогда нужен софтмакс будет
            # основная проблема нулевая ошибка в кроссэнтропии
            y_pred = unet(x)
            loss = loss_func(y_pred, y)
            optm.zero_grad()
            loss.backward()
            optm.step()
        print(loss.item())
        with torch.no_grad():
            history_loss.append(0)
            history_jac_mac.append(0)
            history_jac_mic.append(0)
            for x, y in test_dataloader:
                y_pred = unet(x)
                loss = loss_func(y_pred, y)
                mjma = metric_jaccard_macro(y_pred, y.to(dtype=torch.int32))
                mjmi = metric_jaccard_micro(y_pred, y.to(dtype=torch.int32))
                history_loss[-1] += loss
                history_jac_mac[-1] += mjma
                history_jac_mic[-1] += mjmi
            history_loss[-1] /= length_test_data
            history_jac_mac[-1] /= length_test_data
            history_jac_mic[-1] /= length_test_data
        if ep % 2 == 0:
            torch.save(unet.state_dict(), f"unet_after_{ep}_10_BS")
    return unet, history_loss, history_jac_mac, history_jac_mic


if __name__ == "__main__":
    coco_module = COCO("Datasets/COCO persons/annotations/instances_train2017.json")
    unet_, history_loss_, history_jac_mac_, history_jac_mic_ = main(coco_module)
    # filtered_annotation_ = get_filtered_data(coco_module)
    # filtered_annotation_ = filtered_annotation_[:100]
    # length_train_data_ = int(len(filtered_annotation_) * 0.8)
    # dataset_train_ = IterableDatasetCOCO(coco_module, filtered_annotation_[:length_train_data_],
    #                                      (300, 300), torch.device("cuda"))
    # dataloader_ = DataLoader(dataset_train_, batch_size=3)
    # for i, data in enumerate(dataloader_):
    #     if i == 3:
    #         break
    #     x, y = data
    #     unet_(x)
