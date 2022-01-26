import sys, os, argparse, glob, pathlib, time
import subprocess

import numpy as np
from natsort import natsorted
from tqdm import tqdm
import pdb
from TMmodule import utils, models, io

import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Thinking Microscope parameters')


    input_img_args = parser.add_argument_group("input image arguments")
    input_img_args.add_argument('--look_one_level_down', action='store_true', help='run processing on all subdirectories of current folder')
    input_img_args.add_argument('--dir',
                        default=[], type=str, help='folder containing data to run or train on.')
    input_img_args.add_argument('--mxnet', action='store_true', help='use mxnet')
    input_img_args.add_argument('--all_channels', action='store_true', help='use all channels in image if using own model and images with special channels')

    # model settings 
    model_args = parser.add_argument_group("model arguments")
    parser.add_argument('--pretrained_model', required=False, default='cyto', type=str, help='model to use')
    parser.add_argument('--unet', required=False, default=0, type=int, help='run standard unet instead of cellpose flow output')
    model_args.add_argument('--nclasses',default=3, type=int, help='if running unet, choose 2 or 3; if training omni, choose 4; standard Cellpose uses 3')

    # training settings
    training_args = parser.add_argument_group("training arguments")
    training_args.add_argument('--train', action='store_true', help='train network using images in dir')
    training_args.add_argument('--train_size', action='store_true', help='train size network at end of training')
    training_args.add_argument('--mask_filter',
                        default='_masks', type=str, help='end string for masks to run on. Default: %(default)s')
    training_args.add_argument('--test_dir',
                        default=[], type=str, help='folder containing test data (optional)')
    training_args.add_argument('--learning_rate',
                        default=0.2, type=float, help='learning rate. Default: %(default)s')
    training_args.add_argument('--n_epochs',
                        default=500, type=int, help='number of epochs. Default: %(default)s')
    training_args.add_argument('--batch_size',
                        default=8, type=int, help='batch size. Default: %(default)s')
    training_args.add_argument('--residual_on',
                        default=1, type=int, help='use residual connections')
    training_args.add_argument('--style_on',
                        default=1, type=int, help='use style vector')
    training_args.add_argument('--concatenation',
                        default=0, type=int, help='concatenate downsampled layers with upsampled layers (off by default which means they are added)')
    training_args.add_argument('--save_every',
                        default=100, type=int, help='number of epochs to skip between saves. Default: %(default)s')
    training_args.add_argument('--save_each', action='store_true', help='save the model under a different filename per --save_every epoch for later comparsion')
    
    

    args = parser.parse_args()

    device, gpu = models.assign_device((not args.mxnet), True)

    #define available model names, right now we have three broad categories 
    model_names = ['cyto','nuclei','bact','cyto2','bact_omni','cyto2_omni']
    builtin_model = np.any([args.pretrained_model==s for s in model_names])
    cytoplasmic = 'cyto' in args.pretrained_model
    szmean = 30.
    rescale = True

    # find images
    img_filter = []
    if len(img_filter)>0:
        imf = img_filter
    else:
        imf = None

    image_names = io.get_image_files(args.dir, 
                                             args.mask_filter, 
                                             imf=imf,
                                             look_one_level_down=args.look_one_level_down)
    nimg = len(image_names)


    test_dir = None if len(args.test_dir)==0 else args.test_dir

    output = io.load_train_test_data(args.dir, test_dir, imf, args.mask_filter, args.unet, args.look_one_level_down)
    images, labels, image_names, test_images, test_labels, image_names_test = output

    # training with all channels
    if args.all_channels:
        img = images[0]
        if img.ndim==3:
            nchan = min(img.shape)
        elif img.ndim==2:
            nchan = 1
        channels = None 
    else:
        nchan = 2 

    cpmodel_path = args.pretrained_model

    model = models.CellposeModel(device=device,
                                            torch=(not args.mxnet),
                                            pretrained_model=cpmodel_path, 
                                            diam_mean=szmean,
                                            residual_on=args.residual_on,
                                            style_on=args.style_on,
                                            concatenation=args.concatenation,
                                            nclasses=args.nclasses,
                                            nchan=nchan,
                                            omni=args.omni)

    
    
    

    # Train Segmentation Model
    cpmodel_path = model.train(images, labels, train_files=image_names,
                                           test_data=test_images, test_labels=test_labels, test_files=image_names_test,
                                           learning_rate=args.learning_rate, channels=channels,
                                           save_path=os.path.realpath(args.dir), save_every=args.save_every,
                                           save_each=args.save_each,
                                           rescale=rescale,n_epochs=args.n_epochs,
                                           batch_size=args.batch_size, omni=False)

   
    logger.info('>>>> model trained and saved to %s'%cpmodel_path)

    



if __name__ == '__main__':
    main()