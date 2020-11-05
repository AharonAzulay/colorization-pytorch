

# Train classification network on small training set first
# cp ./checkpoints/siggraph_retrained/latest_net_G.pth ./checkpoints/siggraph_class_small/
python train.py --name siggraph_class_small --sample_p 1.0 --niter 100 --niter_decay 0 --classification --phase train_small --input_nc 5 --output_nc 10 --gpu_ids 0 --load_model --input_nc 5 --output_nc 10 --n_frames 5

# Train classification network first
mkdir ./checkpoints/siggraph_class
cp ./checkpoints/siggraph_class_small/latest_net_G.pth ./checkpoints/siggraph_class/
python train.py --name siggraph_class --sample_p 1.0 --niter 15 --niter_decay 0 --classification --load_model --phase train --gpu_ids 0 --load_model --input_nc 5 --output_nc 10 --n_frames 5


# # Train regression model (with color hints)
# mkdir ./checkpoints/siggraph_reg
# cp ./checkpoints/siggraph_class/latest_net_G.pth ./checkpoints/siggraph_reg/
# python train.py --name siggraph_reg --sample_p .125 --niter 10 --niter_decay 0 --lr 0.00001 --load_model --phase train

# # Turn down learning rate to 1e-6
# mkdir ./checkpoints/siggraph_reg2
# cp ./checkpoints/siggraph_reg/latest_net_G.pth ./checkpoints/siggraph_reg2/
# python train.py --name siggraph_reg2 --sample_p .125 --niter 5 --niter_decay 0 --lr 0.000001 --load_model --phase train
