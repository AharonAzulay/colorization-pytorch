
import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html

import string
import torch
import torchvision
import torchvision.transforms as transforms

from util import util
import numpy as np
import imageio

def arr2vid(arr,video_filename="out",fps = 10):
    print("creating video...")
    height = arr.shape[1]
    width = arr.shape[2]
    w = imageio.get_writer(video_filename + '.mp4', fps=fps)
    for im in arr:
        w.append_data(im)
    w.close()

if __name__ == '__main__':
    sample_ps = [1.]
    to_visualize = ['gray', 'real', 'fake_reg', ]
    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1   # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'test'
    opt.dataroot = './dataset/ilsvrc2012/' + opt.phase
    opt.serial_batches = True
    opt.aspect_ratio = 1.

    dataset = torchvision.datasets.ImageFolder(opt.dataroot,
                                               transform=transforms.Compose([
                                                   transforms.Resize((opt.loadSize, opt.loadSize)),
                                                   transforms.ToTensor()]))
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=not opt.serial_batches)

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

    for i, data_raw in enumerate(dataset_loader):
        data_raw[0] = data_raw[0].to(device)
        data_raw[0] = util.crop_mult(data_raw[0], mult=8)
        data = {}
        A = []
        B = []

        HintB = []
        MaskB = []
        for f in range(0, data_raw[0].shape[1] * opt.n_frames, 3):
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
            arr2vid(vid,"vid_{}".format(i))
            # psnrs[i, pp] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
            entrs[i, pp] = model.get_current_losses()['G_entr']

            # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
            print("")
        if i % 5 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))

        if i == opt.how_many - 1:
            break

    webpage.save()

    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs, axis=0)
    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)

    entrs_mean = np.mean(entrs, axis=0)
    entrs_std = np.std(entrs, axis=0) / np.sqrt(opt.how_many)

    for (pp, sample_p) in enumerate(sample_ps):
        print('p=%.3f: %.2f+/-%.2f' % (sample_p, psnrs_mean[pp], psnrs_std[pp]))
