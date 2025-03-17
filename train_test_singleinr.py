#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime
import torch
import os
import random
import numpy as np
from utils import set_gpu, read_yaml, write_dict_as_yaml, get_mgrid
from model.layer_activations import Sine, WIRE, Relu, FINER
from model.mlp import (MLP, ResMLP, MLPHiddenCoords, ResMLPHiddenCoords, SegmentationHead, ModulatedSiren,
                       ReconstructionHead, SegmentationHeadSDM, Siren)
from model.mapper import MultiResHashGrid, MapperNoEncoding, MapperFF
from training import train_oct_inr_single_octa
from evaluation import eval_oct_inr_single_octa


def parse_args():
    parser = argparse.ArgumentParser(description='INR for Combined OCT and SLO fitting')
    parser.add_argument('config', help='config file (.yaml) containing the hyper-parameters for training.')
    parser.add_argument('--result_dir', type=str, default=None, help='path to directory in which all output directory will be saved.')
    parser.add_argument('--result_note', type=str, default='', help='Directory note to distinguish results.')
    parser.add_argument('--eval', type=str, default='', help='Directory containing model checkpoint.')
    parser.add_argument('--fold', type=int, default=None, help='Set fold number (1-5)')
    parser.add_argument('--torch_compile', action='store_true', default=False, help='Use torch.compile().')
    parser.add_argument('--gpu', default=0, type=int, help="GPU selection.")
    return parser.parse_args()

def main(args):

    # Load the config
    config, config_dict = read_yaml(args.config)

    # bypass some arguments if necessary
    if args.result_dir is not None:
        config.SETTINGS.RESULT_DIR = args.result_dir
        config_dict['SETTINGS']['RESULT_DIR'] = args.result_dir

    if args.fold is not None:
        config.SETTINGS.FOLD = args.fold
        config_dict['SETTINGS']['FOLD'] = args.fold

    if args.result_note:
        args.result_note = '_' + args.result_note

    if args.eval:
        eval_mode = True
    else:
        eval_mode = False

    # set gpu
    set_gpu(int(args.gpu))

    # set seeds for reproducibility
    seed = config.TRAINING.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # maybe redundant (see: https://pytorch.org/docs/stable/notes/randomness.html)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # load train and test data containers
    dataset_train = np.load(os.path.join(config.SETTINGS.IMG_DIR, f'dataset_flat_subsmpl16_train_fold{config.SETTINGS.FOLD}.npz'))
    dataset_subsmpl_train = np.load(os.path.join(config.SETTINGS.IMG_DIR, f'dataset_flat_subsmpl32_train_fold{config.SETTINGS.FOLD}.npz'))
    dataset_test = np.load(os.path.join(config.SETTINGS.IMG_DIR, f'dataset_flat_subsmpl16_test_fold{config.SETTINGS.FOLD}.npz'))
    dataset_subsmpl_test = np.load(os.path.join(config.SETTINGS.IMG_DIR, f'dataset_flat_subsmpl32_test_fold{config.SETTINGS.FOLD}.npz'))

    # unpack containers
    oct_vols_subsmpl_train = torch.cat((torch.from_numpy(dataset_subsmpl_test['oct_vols']).float(), torch.from_numpy(dataset_subsmpl_train['oct_vols']).float()), dim=0)
    oct_seg_vols_subsmpl_train = torch.cat((torch.from_numpy(dataset_subsmpl_test['seg_vols']).float(), torch.from_numpy(dataset_subsmpl_train['seg_vols']).float()), dim=0)
    slo_imgs_subsmpl_train = torch.cat((torch.from_numpy(dataset_subsmpl_test['slo_imgs']).float(), torch.from_numpy(dataset_subsmpl_train['slo_imgs']).float()), dim=0)
    oct_vols_train = torch.cat((torch.from_numpy(dataset_test['oct_vols']).float(), torch.from_numpy(dataset_train['oct_vols']).float()), dim=0)
    oct_seg_vols_train = torch.cat((torch.from_numpy(dataset_test['seg_vols']).float(), torch.from_numpy(dataset_train['seg_vols']).float()), dim=0)
    slo_imgs_train = torch.cat((torch.from_numpy(dataset_test['slo_imgs']).float(), torch.from_numpy(dataset_train['slo_imgs']).float()), dim=0)
    subject_names = list(dataset_subsmpl_test['subject_names']) + list(dataset_subsmpl_train['subject_names'])

    # container organization
    num_samples_train, H, W, _ = oct_vols_subsmpl_train.shape

    # cat only ones for depth dimension --> slice selection mode during training
    coords = torch.cat((get_mgrid((H, W), 2).unsqueeze(0), torch.ones(1, H * W, 1)), -1).cuda()

    #
    # CONFIGURATION OF ENCODING and MODEL
    #

    # select activation function
    if config.MODEL.ACTIVATION == "SIREN":
        activation = Sine
    elif config.MODEL.ACTIVATION == "RELU":
        activation = Relu
    elif config.MODEL.ACTIVATION == "WIRE":
        activation = WIRE
    elif config.MODEL.ACTIVATION == "FINER":
        activation = FINER
    else:
        raise ValueError

    # select position encoding
    if config.POS_ENCODING.TYPE == "None":
        factor = 1.0
        input_size = 3
        if config.MODEL.ACTIVATION == "SIREN":
            factor = torch.pi
        input_mapper = MapperNoEncoding(factor=factor)
    elif config.POS_ENCODING.TYPE == "FF":
        B_gauss = torch.tensor(np.random.normal(scale=config.POS_ENCODING.FF.SCALE, size=(config.POS_ENCODING.FF.MAPPING_SIZE, 3)),
                               dtype=torch.float32).cuda()
        input_mapper = MapperFF(B=B_gauss, factor=config.POS_ENCODING.FF.FACTOR).cuda()
        input_size = 2 * config.POS_ENCODING.FF.MAPPING_SIZE
    elif config.POS_ENCODING.TYPE == "Hashgrid":
        input_mapper = MultiResHashGrid(dim=config.POS_ENCODING.HASHGRID.DIM,
                                        n_levels=config.POS_ENCODING.HASHGRID.N_LEVELS,
                                        n_features_per_level=config.POS_ENCODING.HASHGRID.N_FEATURES_PER_LEVEL,
                                        log2_hashmap_size=config.POS_ENCODING.HASHGRID.LOG2_HASHMAP_SIZE,
                                        base_resolution=config.POS_ENCODING.HASHGRID.BASE_RESOLUTION,
                                        finest_resolution=config.POS_ENCODING.HASHGRID.FINEST_RESOLUTION).cuda()
        input_size = config.POS_ENCODING.HASHGRID.N_LEVELS * config.POS_ENCODING.HASHGRID.N_FEATURES_PER_LEVEL
    else:
        raise ValueError

    # alter input size for additional slo input
    if config.TRAINING.INPUT_SLO:
        input_size += 1

    #
    # select model
    #
    if config.MODEL.SKIP_CONNECTION:
        if config.MODEL.INPUT_COORD_TO_ALL_LAYERS:
            model = ResMLPHiddenCoords
        else:
            model = ResMLP
    else:
        if config.MODEL.INPUT_COORD_TO_ALL_LAYERS:
            model = MLPHiddenCoords
        else:
            model = MLP

    if eval_mode:
        output_path = args.eval
    else:
        time_stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
        if config.TRAINING.INPUT_SLO:
            use_slo = 'SLO'
        else:
            use_slo = ''
        output_path = os.path.join(config.SETTINGS.RESULT_DIR, time_stamp +
                                   f'SingleFitOCT_{model.__name__}_{activation.__name__}_{use_slo}{args.result_note}_fold{config.SETTINGS.FOLD}')

        if not os.path.exists(output_path):
            os.mkdir(output_path)

        # save modified config
        write_dict_as_yaml(config_dict, os.path.join(output_path, "used_config.yaml"))


    for subject_idx in range(num_samples_train):

        output_subject_path = os.path.join(output_path, f"{subject_names[subject_idx]}")
        if not os.path.exists(output_subject_path):
            os.mkdir(output_subject_path)

        subject_name = subject_names[subject_idx]
        oct_vol_subsmpl_train = oct_vols_subsmpl_train[subject_idx].cuda()
        oct_seg_vol_subsmpl_train = oct_seg_vols_subsmpl_train[subject_idx].cuda()
        slo_img_subsmpl_train = slo_imgs_subsmpl_train[subject_idx].cuda()

        backbone = model(input_size,
                         config.TRAINING.LATENT_CODE.SIZE,
                         activation,
                         hidden_size= config.MODEL.HIDDEN_SIZE,
                         num_hidden_layers=config.MODEL.NUM_LAYERS,
                         dropout=config.MODEL.DROPOUT,
                         input_coord_to_all_layers=config.MODEL.INPUT_COORD_TO_ALL_LAYERS,
                         siren_omega_0=config.POS_ENCODING.SIREN.OMEGA_0,
                         wire_scale_0=config.POS_ENCODING.WIRE.SCALE_0,
                         wire_omega_0=config.POS_ENCODING.WIRE.OMEGA_0,
                         finer_omega_0=config.POS_ENCODING.FINER.OMEGA_0,
                         finer_scale_0=config.POS_ENCODING.FINER.SCALE_0,
                         scale_req_grad=config.POS_ENCODING.FINER.SCALE_REQ_GRAD).cuda()

        if args.torch_compile:
            backbone = torch.compile(backbone)

        # todo: change segmentation head for contour data
        segmentation_head = SegmentationHead(config.MODEL.HIDDEN_SIZE, config.MODEL.NUM_CLASSES).cuda()
        reconstruction_head = ReconstructionHead(config.MODEL.HIDDEN_SIZE).cuda()
        if args.torch_compile:
            segmentation_head = torch.compile(segmentation_head)
            reconstruction_head = torch.compile(reconstruction_head)

        #
        # Train of INR for both OCT reconstruction and retina segmentation for each subject
        #

        if not eval_mode:
            train_oct_inr_single_octa(coords, backbone, reconstruction_head, segmentation_head, input_mapper, (oct_vol_subsmpl_train, oct_seg_vol_subsmpl_train, slo_img_subsmpl_train), config, output_subject_path)

        #load trained model
        trained_model = torch.load(os.path.join(output_subject_path, 'model_best.pt'))

        # todo: to be deleted
        if not args.torch_compile:
            for model_part in ['model', 'reconstruction_head', 'segmentation_head']:
                tmp_dict = trained_model[model_part]
                for key in list(tmp_dict.keys()):
                    tmp_dict[key.replace('_orig_mod.', '')] = tmp_dict.pop(key)

        backbone.load_state_dict(trained_model['model'])
        reconstruction_head.load_state_dict(trained_model['reconstruction_head'])
        segmentation_head.load_state_dict(trained_model['segmentation_head'])

        oct_vol_train = oct_vols_train[subject_idx].cuda()
        oct_seg_vol_train = oct_seg_vols_train[subject_idx].cuda()
        slo_img_train = slo_imgs_train[subject_idx].cuda()
        spacing_oct = dataset_train['spacing_oct']
        subject_name = subject_names[subject_idx]

        # Get results per image
        eval_oct_inr_single_octa(coords, backbone, reconstruction_head, segmentation_head, input_mapper,
                              (oct_vol_train, oct_seg_vol_train, slo_img_train), spacing_oct, subject_name,  config, output_subject_path)


if __name__ == '__main__':
    args = parse_args()
    main(args)
