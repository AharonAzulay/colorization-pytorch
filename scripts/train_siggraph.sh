




# Train classification network on small training set first
cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/siggraph_class_small/
python train.py --name siggraph_class_small --sample_p 1.0 --niter 1 --niter_decay 0 --classification --phase train_small --gpu_ids 0 --load_model --n_frames 5 --start_from_single_frame --lr 0.000001

# Train classification network first
mkdir ./checkpoints/siggraph_class
cp ./checkpoints/siggraph_class_small/latest_net_G.pth ./checkpoints/siggraph_class/
python train.py --name siggraph_class --sample_p 1.0 --niter 1 --niter_decay 0 --classification --load_model --phase train --gpu_ids 0 --load_model --n_frames 5 --lr 0.000001



#--name siggraph_class_video --sample_p 1.0 --niter 100 --niter_decay 0 --classification --load_model --phase train --load_model --dataroot "/Users/aazulay/datasets/SkyvideoDataset/clips/" --n_frames 5 --lr 0.000001


# Train regression model (with color hints)
mkdir ./checkpoints/siggraph_reg
cp ./checkpoints/siggraph_class/latest_net_G.pth ./checkpoints/siggraph_reg/
python train.py --name siggraph_reg --sample_p .125 --niter 1 --niter_decay 0 --lr 0.00001 --load_model --phase train --gpu_ids 0

# Turn down learning rate to 1e-6
mkdir ./checkpoints/siggraph_reg2
cp ./checkpoints/siggraph_reg/latest_net_G.pth ./checkpoints/siggraph_reg2/
python train.py --name siggraph_reg2 --sample_p .125 --niter 1 --niter_decay 0 --lr 0.000001 --load_model --phase train --gpu_ids 0


# Train classification network on real video
mkdir ./checkpoints/siggraph_class_video
cp ./checkpoints/siggraph_reg2/latest_net_G.pth ./checkpoints/siggraph_class_video/
python train.py --name siggraph_class_video --sample_p .125 --niter 100 --niter_decay 0 --load_model --phase train --gpu_ids 0 --load_model --dataroot "/isilon/Datasets/Youtube_Sky/clips/" --n_frames 5 --lr 0.000001
