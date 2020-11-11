
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms
from train import VideoDataset,DataLoader
from util import util
import numpy as np
import imageio
from moviepy.editor import ImageSequenceClip

def arr2vid(arr,video_filename="out",fps = 10):
    print("creating video...")
    height = arr.shape[1]
    width = arr.shape[2]
    w = imageio.get_writer(video_filename + '.mp4', fps=fps)
    for im in arr:
        w.append_data(im)
    w.close()


def arr2gif(array, filename, fps=10, scale=1.0):
    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)

    return clip

if __name__ == '__main__':


    sample_ps = [1.]
    to_visualize = ['gray', 'real', 'fake_reg', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()


    tfms = transforms.Compose([transforms.Resize((opt.loadSize, opt.loadSize)), transforms.ToTensor()])
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'test'
    opt.serial_batches = True
    opt.aspect_ratio = 1.
    Data = VideoDataset(root_dir=opt.dataroot, n=opt.n_frames, transform=tfms)
    Loader = DataLoader(Data, batch_size=1, num_workers=int(opt.num_threads), shuffle=True)

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros((opt.how_many, S))
    entrs = np.zeros((opt.how_many, S))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i, data_raw in enumerate(Loader):

        data_raw = data_raw.to(device)
        data_raw = util.crop_mult(data_raw, mult=8)
        data = {}
        A = []
        B = []

        HintB = []
        MaskB = []
        for f in range(0, data_raw.shape[1], 3):
            #                 print(data_raw[:,f:f+3].shape)
            # d = util.get_colorization_data(data_raw[:,f:f+3], opt, p=opt.sample_p)
            d = util.get_colorization_data(data_raw[:,f:f+3], opt, p=opt.sample_p)
            A.append(d["A"])
            B.append(d["B"])
            HintB.append(d["hint_B"])
            MaskB.append(d["mask_B"])
        data["A"] = torch.cat(A, 1)
        data["B"] = torch.cat(B, 1)
        data["hint_B"] = torch.cat(HintB, 1)
        data["mask_B"] = torch.cat(MaskB, 1)


        # with no points
        for (pp, sample_p) in enumerate(sample_ps):
            img_path = [('%08d_%.3f' % (i, sample_p)).replace('.', 'p')]
            # data = util.get_colorization_data(data_raw[0], opt, ab_thresh=0., p=sample_p)

            model.set_input(data)
            model.test(True)  # True means that losses will be computed
            visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)
            vid = []
            for frames in range(0,visuals['fake_reg'].shape[1],3):
                gray = util.tensor2im(visuals['gray'][:,frames:frames+3])
                real = util.tensor2im(visuals['real'][:,frames:frames+3])
                generated = util.tensor2im(visuals['fake_reg'][:,frames:frames+3])
                vid.append(np.vstack((gray,real,generated)))
                psnrs[i, pp] += util.calculate_psnr_np(real, generated) / (visuals['fake_reg'].shape[1]/3)
                #calculate_smoothness_here
            vid = np.array(vid)
            arr2gif(vid,"vid_{}".format(i))
            entrs[i, pp] = model.get_current_losses()['G_entr']

            # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            # print("")
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i == opt.how_many - 1:
            break

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    entrs_mean = np.mean(entrs, axis=0)
    entrs_std = np.std(entrs, axis=0) / np.sqrt(opt.how_many)

    for (pp, sample_p) in enumerate(sample_ps):
        print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))
