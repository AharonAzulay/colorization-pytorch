import time
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import Visualizer
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, join, isdir
import torch
import torchvision
import torchvision.transforms as transforms
from util import util
import numpy as np
import cv2
from PIL import Image


def downsample2_antialiased(X):
    """
    Proper downsampling by a factor of 2 with antialiasing
    :param X:
    :return:
    """
    kernel = np.array([0.06136, 0.24477, 0.38774, 0.24477, 0.06136])
    dst = cv2.sepFilter2D(X, -1, kernel, kernel, anchor=(1, 1), borderType=cv2.BORDER_REFLECT101)
    return dst[::2, ::2]


def resize_helper(X, shape):
    X = X.squeeze()
    while np.all(np.array(X.shape[:2]) >= np.array(shape) * 2):
        X = downsample2_antialiased(X)
    return cv2.resize(X, dsize=tuple(shape[1::-1]), interpolation=cv2.INTER_LINEAR)


def resize_tavi(X, shape):
    if X.ndim == 2 or X.shape[2] <= 4:
        return resize_helper(X, shape)
    # opencv doesn't work on more than 4 channels
    X1 = resize_helper(X[..., :3], shape)
    X2 = resize_helper(X[..., 3:], shape)
    return np.concatenate([X1, X2], axis=2)


class VideoDataset(Dataset):
    def __init__(self, root_dir, n=1, transform=None):
        self.root_dir = root_dir
        self.Vidlist = [f for f in listdir(self.root_dir) if (isfile(join(self.root_dir, f)) and "DS" not in f)]
        self.n_images = len(self.Vidlist)
        self.n = n
        self.transform = transform

    def __len__(self):
        return len(self.Vidlist)

    def __getitem__(self, idx):
        startframe = np.random.randint(0, 100 - self.n)
        img_name = self.Vidlist[idx]
        cap = cv2.VideoCapture(self.root_dir + img_name)
        Frames = []
        c = 0
        while (1):
            c += 1
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if (c > startframe):
                Frames.append(frame)
                if (len(Frames) == self.n):
                    break
        if self.transform is not None:
            for i in range(len(Frames)):
                Frames[i] = self.transform(Image.fromarray(Frames[i]))
        return torch.cat(Frames)


if __name__ == '__main__':

    opt = TrainOptions().parse()
    tfms = transforms.Compose([transforms.RandomChoice([transforms.Resize(opt.loadSize, interpolation=1),
                                                        transforms.Resize(opt.loadSize, interpolation=2),
                                                        transforms.Resize(opt.loadSize, interpolation=3),
                                                        transforms.Resize((opt.loadSize, opt.loadSize),
                                                                          interpolation=1),
                                                        transforms.Resize((opt.loadSize, opt.loadSize),
                                                                          interpolation=2),
                                                        transforms.Resize((opt.loadSize, opt.loadSize),
                                                                          interpolation=3)]),
                               transforms.RandomChoice([transforms.RandomResizedCrop(opt.fineSize, interpolation=1),
                                                        transforms.RandomResizedCrop(opt.fineSize, interpolation=2),
                                                        transforms.RandomResizedCrop(opt.fineSize, interpolation=3)]),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor()])
    opt.dataroot = './dataset/ilsvrc2012/%s/' % opt.phase
    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=tfms)

    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True,
                                                 num_workers=int(opt.num_threads))

    Data = VideoDataset(root_dir="/Users/aazulay/datasets/SkyvideoDataset/clips/", n=opt.n_frames, transform=tfms)
    # Data = VideoDataset(root_dir="/isilon/Datasets/Youtube_Sky/clips/", n=opt.n_frames, transform=tfms)
    Loader = DataLoader(Data, batch_size=2, num_workers=int(opt.num_threads), shuffle=True)

    model = create_model(opt)

    model.setup(opt)
    model.print_networks(True)
    model.save_networks('latest')
    exit()
    visualizer = Visualizer(opt)
    total_steps = 0
    mode = "Imagenet"
    #     mode = "Videos"
    if (mode == "Videos"):
        dataset_size = len(Data)
        print('#training images = %d' % dataset_size)
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            # for i, data in enumerate(dataset):
            for i, data_raw in enumerate(Loader):
                data_raw = data_raw.cpu()
                data = {}
                A = []
                B = []

                HintB = []
                MaskB = []
                for f in range(0, data_raw.shape[1], 3):
                    #                 print(data_raw[:,f:f+3].shape)
                    d = util.get_colorization_data(data_raw[:, f:f + 3], opt, p=opt.sample_p)
                    A.append(d["A"])
                    B.append(d["B"])
                    HintB.append(d["hint_B"])
                    MaskB.append(d["mask_B"])
                data["A"] = torch.cat(A, 1)
                data["B"] = torch.cat(B, 1)
                data["hint_B"] = torch.cat(HintB, 1)
                data["mask_B"] = torch.cat(MaskB, 1)
                # print("data[A].shape")
                # print(data["A"].shape)
                # print("data[B].shape")
                # print(data["B"].shape)
                # print("data[hint_B].shape")
                # print(data["hint_B"].shape)
                # print("data[mask_B].shape")
                # print(data["mask_B"].shape)
                if (data is None):
                    continue

                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    # time to load data
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    # time to do forward&backward
                    t = time.time() - iter_start_time
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.save_networks('latest')

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
    elif (mode == "Imagenet"):
        dataset_size = len(dataset)
        print('#training images = %d' % dataset_size)
        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay):
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            # for i, data in enumerate(dataset):
            for i, data_raw in enumerate(dataset_loader):
                data_raw[0] = data_raw[0].cpu()
                data = {}
                A = []
                B = []

                HintB = []
                MaskB = []
                for f in range(0, data_raw[0].shape[1]*opt.n_frames, 3):
                    #                 print(data_raw[:,f:f+3].shape)
                    d = util.get_colorization_data(data_raw[0], opt, p=opt.sample_p)
                    A.append(d["A"])
                    B.append(d["B"])
                    HintB.append(d["hint_B"])
                    MaskB.append(d["mask_B"])
                data["A"] = torch.cat(A, 1)
                data["B"] = torch.cat(B, 1)
                data["hint_B"] = torch.cat(HintB, 1)
                data["mask_B"] = torch.cat(MaskB, 1)
                # print("data[A].shape")
                # print(data["A"].shape)
                # print("data[B].shape")
                # print(data["B"].shape)
                # print("data[hint_B].shape")
                # print(data["hint_B"].shape)
                # print("data[mask_B].shape")
                # print(data["mask_B"].shape)
                if (data is None):
                    continue

                iter_start_time = time.time()
                if total_steps % opt.print_freq == 0:
                    # time to load data
                    t_data = iter_start_time - iter_data_time
                visualizer.reset()
                total_steps += opt.batch_size
                epoch_iter += opt.batch_size
                model.set_input(data)
                model.optimize_parameters()

                if total_steps % opt.display_freq == 0:
                    save_result = total_steps % opt.update_html_freq == 0
                    visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                if total_steps % opt.print_freq == 0:
                    losses = model.get_current_losses()
                    # time to do forward&backward
                    t = time.time() - iter_start_time
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data)
                    if opt.display_id > 0:
                        visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, opt, losses)

                if total_steps % opt.save_latest_freq == 0:
                    print('saving the latest model (epoch %d, total_steps %d)' %
                          (epoch, total_steps))
                    model.save_networks('latest')

                iter_data_time = time.time()

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))
                model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate()
