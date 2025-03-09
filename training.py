#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from utils import turn_to_onehot, get_train_indices, AverageMeter, EarlyStopping, SSIMLoss, COMBILoss
from evaluation import compute_smooth_dice
from random import shuffle
from plots import plot_step_summary, plot_step_summary_oct_with_seg, plot_step_summary_oct_with_seg_single


def train_oct_inr_single_octa(coords, model, reconstruction_head, segmentation_head, input_mapper, oct_seg_slo, config, output_path):

    print('Loading training data...')
    oct_vol, oct_seg_vol, slo_img = oct_seg_slo

    H, W, D = oct_vol.shape
    num_classes = config.MODEL.NUM_CLASSES

    # set range for depth dimension
    # Division by 10 since z-spacing is roughly 10x x-spacing --> pretending isotropic spacing to allow for interpolation
    d_range = (torch.linspace(-1, 1, D) / 10).cuda()
    gt_oct_center = oct_vol[..., D // 2]
    gt_oct_center = gt_oct_center.reshape(1, -1, 1).cuda()

    gt_seg_center = turn_to_onehot(oct_seg_vol[..., D // 2], num_classes).reshape(1, -1, num_classes).cuda()
    slo_input_center = slo_img[D // 2, :]
    slo_input_center = slo_input_center.reshape(1, -1, 1).repeat((1, H, 1)).cuda()

    coord_input_center = coords * torch.tensor([1, 1, d_range[D // 2]]).view(1, 1, 3).cuda()

    lr = config.TRAINING.LEARNING_RATE
    total_epochs = config.TRAINING.TOTAL_EPOCHS
    epochs_to_summary = config.TRAINING.EPOCHS_TIL_SUMMARY

    optim_params = [{'params': model.parameters(), 'lr': lr},
                    {'params': reconstruction_head.parameters(), 'lr': lr},
                    {'params': segmentation_head.parameters(), 'lr': lr}]

    if config.POS_ENCODING.TYPE == 'Hashgrid':
        optim_params.append({'params': input_mapper.parameters(), 'lr': lr})

    optim = torch.optim.Adam(optim_params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    if config.TRAINING.LOSS_RECON == 'MSE':
        criterion_recon = nn.MSELoss()
    elif config.TRAINING.LOSS_RECON == 'SSIM':
        criterion_recon = SSIMLoss()
    elif config.TRAINING.LOSS_RECON == 'Combi':
        criterion_recon = COMBILoss()
    else:
        raise ValueError

    criterion_seg = nn.BCELoss()

    all_losses = []
    all_dices = []
    best_loss = np.inf
    best_step = np.nan

    early_stopping = EarlyStopping(patience=20, min_delta=0.00001, min_epochs=100)

    #TODO: change this to single
    vol_indices = np.arange(D)

    epoch_time = AverageMeter()
    end = time.time()
    for epoch in range(total_epochs):

        shuffle(vol_indices)
        losses = AverageMeter()
        losses_recon = AverageMeter()
        losses_seg = AverageMeter()
        dices = AverageMeter()


        for iter, idx in enumerate(vol_indices):

            model.train()
            segmentation_head.train()
            reconstruction_head.train()

            # select all intensities and reshape them
            gt_oct = oct_vol[..., idx]
            gt_oct = gt_oct.reshape(1, -1, 1).cuda()

            gt_seg = turn_to_onehot(oct_seg_vol[..., idx], num_classes).reshape(1, -1, num_classes).cuda()

            coord_input = coords * torch.tensor([1, 1, d_range[idx]]).view(1, 1, 3).cuda()
            slo_input = slo_img[idx, ...]
            slo_input = slo_input.reshape(1, -1, 1).repeat((1, H, 1)).cuda()

            _, N, _ = coord_input.shape

            # forward step
            if config.TRAINING.INPUT_SLO:
                output_backbone = model((torch.cat([input_mapper(coord_input), slo_input], dim=2), torch.zeros(1, N, 0).cuda()))
            else:
                output_backbone = model((input_mapper(coord_input), torch.zeros(1, N, 0).cuda()))
            output_recon = reconstruction_head(output_backbone)
            output_seg = segmentation_head(output_backbone)

            # loss computation
            loss_recon = criterion_recon(output_recon.view(1, 1, H, W), gt_oct.view(1, 1, H, W))
            loss_seg = criterion_seg(output_seg, gt_seg)

            loss = float(config.TRAINING.LOSS_WEIGHT_RECON) * loss_recon + float(config.TRAINING.LOSS_WEIGHT_SEG) * loss_seg

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.update(loss.item(), 1)
            losses_recon.update(loss_recon.item(), 1)
            losses_seg.update(loss_seg.item(), 1)
            dices.update(compute_smooth_dice(output_seg.detach().cpu(), gt_seg.detach().cpu()), 1)

        epoch_time.update(time.time() - end)
        end = time.time()

        print(f"[TRAIN] Epoch {epoch + 1:3} - loss: {losses.avg:.4f} - loss_recon: {losses_recon.avg:.4f} - loss_seg: {losses_seg.avg:.4f} - dice: {dices.avg:.4f} - ET {epoch_time.val:.3f} ({epoch_time.avg:.3f}) s")

        scheduler.step()
        all_losses.append(losses.avg)
        all_dices.append(dices.avg)

        model.eval()
        reconstruction_head.eval()
        segmentation_head.eval()

        with torch.no_grad():

            if config.TRAINING.INPUT_SLO:
                output_backbone_center = model(
                    (torch.cat([input_mapper(coord_input_center), slo_input_center], dim=2), torch.zeros(1, N, 0).cuda()))
            else:
                output_backbone_center = model((input_mapper(coord_input_center), torch.zeros(1, N, 0).cuda()))
            output_recon_center = reconstruction_head(output_backbone_center)
            output_seg_center = segmentation_head(output_backbone_center)

        early_stopping(epoch, losses.avg)

        curr_loss = losses.avg
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_step = epoch
            state = {'model': model.state_dict(),
                     'reconstruction_head': reconstruction_head.state_dict(),
                     'segmentation_head': segmentation_head.state_dict(),
                     'step': epoch}
            torch.save(state, os.path.join(output_path, 'model_best.pt'))

        if epoch == 0 or (epoch + 1) % epochs_to_summary == 0 or epoch == total_epochs - 1 or early_stopping.early_stop:

            model_output_oct_center = output_recon_center.detach().cpu().view((H, W)).numpy()
            model_output_seg_center = torch.argmax(output_seg_center.view((H, W, num_classes)), dim=-1).detach().cpu().numpy()

            plot_step_summary_oct_with_seg_single(model_output_oct_center,
                                                  gt_oct_center.cpu().view((H, W)).numpy(),
                                                  model_output_seg_center,
                                                  torch.argmax(gt_seg_center.cpu().view((H, W, num_classes)), dim=-1).numpy(),
                                                  all_losses,
                                                  output_path,
                                                  epoch)

            state = {'model': model.state_dict(),
                     'reconstruction_head': reconstruction_head.state_dict(),
                     'segmentation_head': segmentation_head.state_dict(),
                     'step': epoch}

            torch.save(state, os.path.join(output_path, 'model_last.pt'))

        if early_stopping.early_stop:
            break

    print('Best loss {:.6f} achieved in step {:.0f}.'.format(best_loss, best_step))


def train_oct_inr_gen_octa(coords, latent_codes, model, reconstruction_head, segmentation_head, input_mapper, dataset_subsmpl_train, config, output_path, lc_fit=False):

    print('Loading training data...')
    if not lc_fit:
        oct_vols = torch.from_numpy(dataset_subsmpl_train['oct_vols']).float().cuda()
        oct_seg_vols = torch.from_numpy(dataset_subsmpl_train['seg_vols']).float().cuda()
        slo_imgs = torch.from_numpy(dataset_subsmpl_train['slo_imgs']).float().cuda()
        #todo: use boundary data
    else:
        oct_vols, oct_seg_vols, slo_imgs = dataset_subsmpl_train

    print('done.')

    num_samples, H, W, D = oct_vols.shape
    num_classes = config.MODEL.NUM_CLASSES

    for idx_smpl in range(num_samples):
        oct = oct_vols[idx_smpl]
        oct = normalize(oct, new_min=0, new_max=1)
        oct_vols[idx_smpl] = oct

        slo = slo_imgs[idx_smpl]
        slo = normalize(slo, new_min=0, new_max=1)
        slo_imgs[idx_smpl] = slo

    # set range for depth dimension
    # Division by 10 since z-spacing is roughly 10x x-spacing --> pretending isotropic spacing to allow for interpolation
    d_range = (torch.linspace(-1, 1, D) / 10).cuda()
    gt_oct_center = oct_vols[0][..., D // 2]
    gt_oct_center = gt_oct_center.reshape(1, -1, 1).cuda()

    gt_seg_center = turn_to_onehot(oct_seg_vols[0][..., D // 2], num_classes).reshape(1, -1, num_classes).cuda()
    slo_input_center = slo_imgs[0][D // 2, :]
    slo_input_center = slo_input_center.reshape(1, -1, 1).repeat((1, H, 1)).cuda()

    coord_input_center = coords * torch.tensor([1, 1, d_range[D // 2]]).view(1, 1, 3).cuda()

    if not lc_fit:
        lr = config.TRAINING.LEARNING_RATE
        total_epochs = config.TRAINING.TOTAL_EPOCHS
        epochs_to_summary = config.TRAINING.EPOCHS_TIL_SUMMARY
    else:
        lr = config.TEST.LEARNING_RATE
        total_epochs = config.TEST.TOTAL_EPOCHS
        epochs_to_summary = config.TEST.EPOCHS_TIL_SUMMARY


    optim_params = [{'params': latent_codes, 'lr': lr}]

    if not lc_fit:
        optim_params += [{'params': model.parameters(), 'lr': lr},
                         {'params': reconstruction_head.parameters(), 'lr': lr},
                         {'params': segmentation_head.parameters(), 'lr': lr}]

    if config.POS_ENCODING.TYPE == 'Hashgrid':
        optim_params.append({'params': input_mapper.parameters(), 'lr': lr})

    optim = torch.optim.Adam(optim_params)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.99)

    if config.TRAINING.LOSS_RECON == 'MSE':
        criterion_recon = nn.MSELoss()
    elif config.TRAINING.LOSS_RECON == 'SSIM':
        criterion_recon = SSIMLoss()
    elif config.TRAINING.LOSS_RECON == 'Combi':
        criterion_recon = COMBILoss()
    else:
        raise ValueError

    criterion_seg = nn.BCELoss()

    all_losses = []
    all_dices = []
    best_loss = np.inf
    best_step = np.nan

    early_stopping = EarlyStopping(patience=20, min_delta=0.00001, min_epochs=100)

    h_init = latent_codes.clone().detach().cpu()

    all_indices = get_train_indices(num_samples, D)

    epoch_time = AverageMeter()
    end = time.time()
    for epoch in range(total_epochs):

        shuffle(all_indices)
        losses = AverageMeter()
        losses_recon = AverageMeter()
        losses_seg = AverageMeter()
        dices = AverageMeter()


        for iter, (idx_smpl, idx) in enumerate(all_indices):

            model.train()
            segmentation_head.train()
            reconstruction_head.train()

            # select all intensities and reshape them
            gt_oct = oct_vols[idx_smpl][..., idx]
            gt_oct = gt_oct.reshape(1, -1, 1).cuda()

            gt_seg = turn_to_onehot(oct_seg_vols[idx_smpl][..., idx], num_classes).reshape(1, -1, num_classes).cuda()

            coord_input = coords * torch.tensor([1, 1, d_range[idx]]).view(1, 1, 3).cuda()
            slo_input = slo_imgs[idx_smpl][idx, ...]
            slo_input = slo_input.reshape(1, -1, 1).repeat((1, H, 1)).cuda()

            _, N, _ = coord_input.shape

            if config.MODEL.MODULATION:
                h = latent_codes[idx_smpl, ...].cuda()
            else:
                h = latent_codes[idx_smpl, ...].tile(1, N, 1).cuda()

            # forward step
            if config.TRAINING.INPUT_SLO:
                output_backbone = model((torch.cat([input_mapper(coord_input), slo_input], dim=2), h))
            else:
                output_backbone = model((input_mapper(coord_input), h))
            output_recon = reconstruction_head(output_backbone)
            output_seg = segmentation_head(output_backbone)

            # loss computation
            loss_recon = criterion_recon(output_recon.view(1, 1, H, W), gt_oct.view(1, 1, H, W))
            loss_seg = criterion_seg(output_seg, gt_seg)
            loss_reg_h = F.mse_loss(h, torch.zeros_like(h))
            if lc_fit:
                loss = float(config.TRAINING.LOSS_WEIGHT_RECON) * loss_recon + loss_reg_h
            else:
                loss = float(config.TRAINING.LOSS_WEIGHT_RECON) * loss_recon + float(config.TRAINING.LOSS_WEIGHT_SEG) * loss_seg + loss_reg_h

            optim.zero_grad()
            loss.backward()
            optim.step()

            losses.update(loss.item(), 1)
            losses_recon.update(loss_recon.item(), 1)
            losses_seg.update(loss_seg.item(), 1)
            dices.update(compute_smooth_dice(output_seg.detach().cpu(), gt_seg.detach().cpu()), 1)

        epoch_time.update(time.time() - end)
        end = time.time()

        print(f"[TRAIN] Epoch {epoch + 1:3} - loss: {losses.avg:.4f} - loss_recon: {losses_recon.avg:.4f} - loss_seg: {losses_seg.avg:.4f} - dice: {dices.avg:.4f} - ET {epoch_time.val:.3f} ({epoch_time.avg:.3f}) s")

        scheduler.step()
        all_losses.append(losses.avg)
        all_dices.append(dices.avg)

        model.eval()
        reconstruction_head.eval()
        segmentation_head.eval()

        with torch.no_grad():
            if config.MODEL.MODULATION:
                h = latent_codes[0, ...].cuda()
            else:
                h = latent_codes[0, ...].tile(1, N, 1).cuda()
            if config.TRAINING.INPUT_SLO:
                output_backbone_center = model(
                    (torch.cat([input_mapper(coord_input_center), slo_input_center], dim=2), h))
            else:
                output_backbone_center = model((input_mapper(coord_input_center), h))
            output_recon_center = reconstruction_head(output_backbone_center)
            output_seg_center = segmentation_head(output_backbone_center)

        if not lc_fit:
            early_stopping(epoch, losses.avg)

        curr_loss = losses.avg
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_step = epoch
            state = {'latent_codes': latent_codes,
                     'step': epoch}
            if not lc_fit:
                state.update({'model': model.state_dict(),
                              'reconstruction_head': reconstruction_head.state_dict(),
                              'segmentation_head': segmentation_head.state_dict()
                              })
            torch.save(state, os.path.join(output_path, 'model_best.pt'))

        if epoch == 0 or (epoch + 1) % epochs_to_summary == 0 or epoch == total_epochs - 1 or early_stopping.early_stop:

            model_output_oct_center = output_recon_center.detach().cpu().view((H, W)).numpy()
            model_output_seg_center = torch.argmax(output_seg_center.view((H, W, num_classes)), dim=-1).detach().cpu().numpy()

            plot_step_summary_oct_with_seg(model_output_oct_center,
                                           gt_oct_center.cpu().view((H, W)).numpy(),
                                           model_output_seg_center,
                                           torch.argmax(gt_seg_center.cpu().view((H, W, num_classes)), dim=-1).numpy(),
                                           all_losses,
                                           h_init,
                                           latent_codes.detach().cpu(),
                                           output_path,
                                           epoch)

            state = {'latent_codes': latent_codes,
                     'step': epoch}
            if not lc_fit:
                state.update({'model': model.state_dict(),
                              'reconstruction_head': reconstruction_head.state_dict(),
                              'segmentation_head': segmentation_head.state_dict()
                              })
            torch.save(state, os.path.join(output_path, 'model_last.pt'))

        if early_stopping.early_stop:
            break

    print('Best loss {:.6f} achieved in step {:.0f}.'.format(best_loss, best_step))