#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import pickle
import lpips
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
from data_utils import save_as_itk, normalize
from plots import plot_final_summary, plot_final_summary_oct
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from utils import turn_to_onehot, AverageMeter

def calculate_mae(ref_img, img):
    mae = np.mean(np.abs(ref_img - img))
    return mae


def calculate_mse(ref_img, img):
    mse = np.mean((ref_img - img) ** 2)
    return mse


def calculate_psnr(ref_img, img):
    psnr = peak_signal_noise_ratio(ref_img, img, data_range=ref_img.max()-ref_img.min())
    return psnr


def calculate_ssim(ref_img, img):
    ssim = structural_similarity(ref_img, img, data_range=ref_img.max()-ref_img.min())
    return ssim


def calculate_lpips(ref_img, img, mode, device):
    '''
    Based on: https://github.com/jqmcginnis/multi_contrast_inr/blob/main/utils.py
    TODO: Handling of 3D images - Code found online calculates LPIPS-Loss for each image slice in each direction,
    but for OCT images with only 25 B-Scans inputs are too small to be passed through AlexNet
    Images with 49 B-Scans do work, but maybe it would be better to calculate the loss for 2D B-Scans only, anyways
    '''
    dimensionality = len(ref_img.shape)
    ref_ = torch.tensor(ref_img).float()
    img_ = torch.tensor(img).float()

    lpips_loss = lpips.LPIPS(net='alex')

    if dimensionality == 2:
        H, W = ref_img.shape
        ref_t = ref_.reshape(1, H, W).repeat(3, 1, 1).to(device)
        pred_t = img_.reshape(1, H, W).repeat(3, 1, 1).to(device)
        pips = lpips_loss(ref_t, pred_t)

    elif dimensionality == 3:
        H, W, D = ref_img.shape
        pips = 0

        if mode == '2D':
            for d in range(D):
                ref_t = ref_[:, :, d].reshape(1, H, W).repeat(3, 1, 1).to(device)
                pred_t = img_[:, :, d].reshape(1, H, W).repeat(3, 1, 1).to(device)
                pips += lpips_loss(ref_t, pred_t)
            pips /= D

        elif mode == '3D':
            for d in range(D):
                ref_t = ref_[:, :, d].reshape(1, H, W).repeat(3, 1, 1).to(device)
                pred_t = img_[:, :, d].reshape(1, H, W).repeat(3, 1, 1).to(device)
                pips += lpips_loss(ref_t, pred_t)

            for h in range(H):
                ref_t = ref_[h, :, :].reshape(1, W, D).repeat(3, 1, 1).to(device)
                pred_t = img_[h, :, :].reshape(1, W, D).repeat(3, 1, 1).to(device)
                pips += lpips_loss(ref_t, pred_t)

            for w in range(W):
                ref_t = ref_[:, w, :].reshape(1, H, D).repeat(3, 1, 1).to(device)
                pred_t = img_[:, w, :].reshape(1, H, D).repeat(3, 1, 1).to(device)
                pips += lpips_loss(ref_t, pred_t)

            pips /= (H + W + D)

        else:
            raise ValueError('Mode must be either 2D or 3D!')

    else:
        raise ValueError('Input images must be either 2D or 3D!')

    return pips.item()


def compute_smooth_dice(y_pred, y_true, eps=1e-12):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    return (2*y_pred*y_true+eps).sum()/((y_pred.abs()+y_true.abs()).sum()+eps)


def calculate_eval_metrics(gt_img, recon_img, mode='2D', device='cpu'):
    # Mode can be either 2D or 3D (only relevant for 3D inputs)
    # 2D: Metrics are calculated per B-Scan and then averaged
    # 3D: Metrics are calculated on the entire image volume

    metrics = {}

    dimensionality = len(gt_img.shape)
    if dimensionality == 2:
        print('Input images are 2D. Ignoring parameter mode.')
        metrics['MSE'] = calculate_mse(gt_img, recon_img)
        metrics['MAE'] = calculate_mae(gt_img, recon_img)
        metrics['PSNR'] = calculate_psnr(gt_img, recon_img)
        metrics['SSIM'] = calculate_ssim(gt_img, recon_img)

    elif dimensionality == 3:
        print('Using {} mode for calculation of metrics.'.format(mode))
        if mode == '3D':
            metrics['MSE'] = calculate_mse(gt_img, recon_img)
            metrics['MAE'] = calculate_mae(gt_img, recon_img)
            metrics['PSNR'] = calculate_psnr(gt_img, recon_img)
            metrics['SSIM'] = calculate_ssim(gt_img, recon_img)
        elif mode == '2D':
            H, W, D = gt_img.shape
            mse = 0
            mae = 0
            psnr = 0
            ssim = 0
            for d in range(D):
                mse += calculate_mse(gt_img[:, :, d], recon_img[:, :, d])
                mae += calculate_mae(gt_img[:, :, d], recon_img[:, :, d])
                psnr += calculate_psnr(gt_img[:, :, d], recon_img[:, :, d])
                ssim += calculate_ssim(gt_img[:, :, d], recon_img[:, :, d])

            mse /= D
            mae /= D
            psnr /= D
            ssim /= D

            metrics['MSE'] = mse
            metrics['MAE'] = mae
            metrics['PSNR'] = psnr
            metrics['SSIM'] = ssim

    if dimensionality == 2 or mode == '3D':
        if np.all(np.array(gt_img.shape) > 30):
            metrics['LPIPS'] = calculate_lpips(gt_img, recon_img, mode, device)
        else:
            print('LPIPS metric can not be calculated. Image size too small.')
    elif dimensionality == 3 and mode == '2D':
        H, W, _ = gt_img.shape
        if H > 30 and W > 30:
            metrics['LPIPS'] = calculate_lpips(gt_img, recon_img, mode, device)
        else:
            print('LPIPS metric can not be calculated. Image size too small.')
    for name, value in metrics.items():
        print('{}: {:.6f}'.format(name, value))
    print('\n')
    return metrics


def eval_oct_inr_gen_octa(coords, latent_codes, model, reconstruction_head, segmentation_head, input_mapper, dataset_flat_eval, config, output_path, lc_fit=False):

    print("Load evaluation data...")
    if not lc_fit:
        oct_vols_eval = torch.from_numpy(dataset_flat_eval['oct_vols']).float().cuda()
        oct_seg_vols_eval = torch.from_numpy(dataset_flat_eval['seg_vols']).float().cuda()
        slo_imgs_eval = torch.from_numpy(dataset_flat_eval['slo_imgs']).float().cuda()
        subject_names = dataset_flat_eval['subject_names']
        spacing_oct = dataset_flat_eval['spacing_oct']
    else:
        oct_vols_eval, oct_seg_vols_eval, slo_imgs_eval, spacing_oct, subject_names = dataset_flat_eval

    num_samples_eval = oct_vols_eval.shape[0]
    H, W, D = oct_vols_eval[0].shape
    num_classes = config.MODEL.NUM_CLASSES


    for idx_smpl in range(num_samples_eval):
        oct = oct_vols_eval[idx_smpl]
        oct = normalize(oct, new_min=0, new_max=1)
        oct_vols_eval[idx_smpl] = oct

        slo = slo_imgs_eval[idx_smpl]
        slo = normalize(slo, new_min=0, new_max=1)
        slo_imgs_eval[idx_smpl] = slo

    # set range for depth dimension
    # Division by 10 since z-spacing is roughly 10x x-spacing --> pretending isotropic spacing to allow for interpolation
    d_range = torch.linspace(-1, 1, D) / 10

    # todo: change config and used data such that subsampling factor corresponds to actually used training data
    idx_train = [i for i in range(len(d_range)) if i % config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR == 0]
    idx_interp = [i for i in range(len(d_range)) if i % config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR != 0]
    idx_train = np.arange(0, 31, 2)
    idx_interp = np.arange(1, 31, 2)

    criterion_recon = nn.MSELoss()
    criterion_seg = nn.BCELoss()
    recon_losses = []
    seg_losses = []

    model.eval()
    segmentation_head.eval()
    reconstruction_head.eval()

    with torch.no_grad():

        for idx_smpl in np.arange(num_samples_eval):
            for idx in np.arange(D):
                gt_oct = oct_vols_eval[idx_smpl][..., idx]
                oct_vols_eval[idx_smpl][..., idx] = gt_oct

                slo_input = slo_imgs_eval[idx_smpl][idx, :]
                slo_imgs_eval[idx_smpl][idx, :] = slo_input

        for idx_smpl in np.arange(num_samples_eval):
            losses_recon = AverageMeter()
            losses_seg = AverageMeter()

            pred_oct_vol = np.zeros((H, W, D))
            pred_seg_vol = np.zeros((H, W, D))

            for idx in np.arange(D):
                gt_oct = oct_vols_eval[idx_smpl][..., idx].reshape(1, -1, 1).cuda()
                gt_seg = turn_to_onehot(oct_seg_vols_eval[idx_smpl][..., idx], num_classes).reshape(1, -1, num_classes).cuda()
                coord_input = coords * torch.tensor([1, 1, d_range[idx]]).view(1, 1, 3).cuda()
                slo_input = slo_imgs_eval[idx_smpl][idx, ...].reshape(1, -1, 1).repeat((1, H, 1)).cuda()

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

                # output_backbone = model((input_mapper(model_input), h))
                output_recon = reconstruction_head(output_backbone)
                output_seg = segmentation_head(output_backbone)
                #print(output_recon.min(), output_recon.max())

                # loss computation
                loss_recon = criterion_recon(output_recon.view(1, 1, H, W), gt_oct.view(1, 1, H, W))
                loss_seg = criterion_seg(output_seg, gt_seg)

                losses_recon.update(loss_recon.item(), 1)
                losses_seg.update(loss_seg.item(), 1)

                pred_oct_vol[..., idx] = output_recon.view((H, W)).detach().cpu().numpy()
                pred_seg_vol[..., idx] = torch.argmax(output_seg.view((H, W, num_classes)), dim=-1).detach().cpu().numpy()

            print("[EVAL] Sample {:3} - loss_recon: {:.4f} - loss_seg: {:.4f} ".format(idx_smpl+1, losses_recon.avg, losses_seg.avg))
            recon_losses.append(losses_recon.avg)
            seg_losses.append(losses_seg.avg)

            oct_vol = oct_vols_eval[idx_smpl].cpu().numpy()

            nifti_result_path = os.path.join(output_path, 'nifti')
            if not os.path.exists(nifti_result_path):
                os.makedirs(nifti_result_path)

            save_as_itk(pred_oct_vol.transpose(2, 0, 1),
                        os.path.join(nifti_result_path, f'{subject_names[idx_smpl]}_vol.nii.gz'), spacing_oct)
            save_as_itk(pred_seg_vol.transpose(2, 0, 1),
                        os.path.join(nifti_result_path, f'{subject_names[idx_smpl]}_vol_seg.nii.gz'), spacing_oct)


            #todo: Save pred_oct_vol and pred_seg_vol!!!

            # plot summary
            plot_final_summary_oct(pred_oct_vol, oct_vol, output_path, idx_smpl=idx_smpl)

            metric_results_path = os.path.join(output_path, 'metric_results')
            if not os.path.exists(metric_results_path):
                os.mkdir(metric_results_path)

            # Evaluate performance for reconstructed and interpolated B-Scans
            print('\nEvaluating reconstruction of OCT B-scans:')
            metrics = calculate_eval_metrics(oct_vol[..., idx_train], pred_oct_vol[..., idx_train], mode='2D')
            with open(os.path.join(metric_results_path, 'recon_results_OCT_smpl{}.pkl'.format(idx_smpl)), 'wb') as f:
                pickle.dump(metrics, f)

            print('Evaluating interpolated OCT B-scans:')
            metrics = calculate_eval_metrics(oct_vol[..., idx_interp], pred_oct_vol[..., idx_interp], mode='2D')
            with open(os.path.join(metric_results_path, 'interp_results_OCT_smpl{}.pkl'.format(idx_smpl)), 'wb') as f:
                pickle.dump(metrics, f)

            n = config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR + 1
            n = 3
            interp_results_path = os.path.join(output_path, 'interpolation_results')
            if not os.path.exists(interp_results_path):
                os.mkdir(interp_results_path)

            for d in range(len(idx_train) - 1):
                fig, ax = plt.subplots(2, n, figsize=(4 * n, 6))
                for idx in range(n):
                    oct_slice = oct_vol[..., d * (n - 1) + idx]
                    ax[0, idx].imshow(oct_slice, cmap='gray')
                    ax[0, idx].title.set_text('GT slice ' + str(d * (n - 1) + idx))
                    ax[1, idx].imshow(pred_oct_vol[..., d * (n - 1) + idx], cmap='gray', vmin=oct_slice.min(),
                                      vmax=oct_slice.max())
                    if idx == 0 or idx == n - 1:
                        ax[1, idx].title.set_text('Recon. slice ' + str(d * (n - 1) + idx))
                    else:
                        ax[1, idx].title.set_text('Interp. slice ' + str(d * (n - 1) + idx))
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

                plt.savefig(os.path.join(interp_results_path, 'interpResult_smpl{}_slice{}.png'.format(idx_smpl, d)))
                plt.close()

        print("[EVAL] Mean over all samples - loss_recon: {:.6f} - loss_seg: {:.6f} ".format(np.array(recon_losses).mean(),
                                                                                             np.array(seg_losses).mean()))


def eval_oct_inr_single_octa(coords, model, reconstruction_head, segmentation_head, input_mapper, oct_seg_slo, spacing, subject_name, config, output_path):

    print("Load evaluation data...")

    oct_vol_eval, oct_seg_vol_eval, slo_img_eval = oct_seg_slo

    H, W, D = oct_vol_eval.shape
    num_classes = config.MODEL.NUM_CLASSES

    # set range for depth dimension
    # Division by 10 since z-spacing is roughly 10x x-spacing --> pretending isotropic spacing to allow for interpolation
    d_range = torch.linspace(-1, 1, D) / 10

    # todo: change config and used data such that subsampling factor corresponds to actually used training data
    idx_train = [i for i in range(len(d_range)) if i % config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR == 0]
    idx_interp = [i for i in range(len(d_range)) if i % config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR != 0]
    idx_train = np.arange(0, 31, 2)
    idx_interp = np.arange(1, 31, 2)

    criterion_recon = nn.MSELoss()
    criterion_seg = nn.BCELoss()
    recon_losses = []
    seg_losses = []

    model.eval()
    segmentation_head.eval()
    reconstruction_head.eval()

    with torch.no_grad():

        for idx in np.arange(D):
            gt_oct = oct_vol_eval[..., idx]
            oct_vol_eval[..., idx] = gt_oct

            slo_input = slo_img_eval[idx, :]
            slo_img_eval[idx, :] = slo_input

        losses_recon = AverageMeter()
        losses_seg = AverageMeter()

        pred_oct_vol = np.zeros((H, W, D))
        pred_seg_vol = np.zeros((H, W, D))

        for idx in np.arange(D):
            gt_oct = oct_vol_eval[..., idx].reshape(1, -1, 1).cuda()
            gt_seg = turn_to_onehot(oct_seg_vol_eval[..., idx], num_classes).reshape(1, -1, num_classes).cuda()
            coord_input = coords * torch.tensor([1, 1, d_range[idx]]).view(1, 1, 3).cuda()
            slo_input = slo_img_eval[idx, ...].reshape(1, -1, 1).repeat((1, H, 1)).cuda()

            _, N, _ = coord_input.shape

            # forward step
            if config.TRAINING.INPUT_SLO:
                output_backbone = model((torch.cat([input_mapper(coord_input), slo_input], dim=2), torch.zeros(1, N, 0).cuda()))
            else:
                output_backbone = model((input_mapper(coord_input), torch.zeros(1, N, 0).cuda()))

            # output_backbone = model((input_mapper(model_input), h))
            output_recon = reconstruction_head(output_backbone)
            output_seg = segmentation_head(output_backbone)
            #print(output_recon.min(), output_recon.max())

            # loss computation
            loss_recon = criterion_recon(output_recon.view(1, 1, H, W), gt_oct.view(1, 1, H, W))
            loss_seg = criterion_seg(output_seg, gt_seg)

            losses_recon.update(loss_recon.item(), 1)
            losses_seg.update(loss_seg.item(), 1)

            pred_oct_vol[..., idx] = output_recon.view((H, W)).detach().cpu().numpy()
            pred_seg_vol[..., idx] = torch.argmax(output_seg.view((H, W, num_classes)), dim=-1).detach().cpu().numpy()

            print(f"[EVAL] {subject_name} loss_recon: {losses_recon.avg:.4f} - loss_seg: {losses_seg.avg:.4f}")
            recon_losses.append(losses_recon.avg)
            seg_losses.append(losses_seg.avg)

            oct_vol = oct_vol_eval.cpu().numpy()

            nifti_result_path = os.path.join(output_path, 'nifti')
            if not os.path.exists(nifti_result_path):
                os.makedirs(nifti_result_path)


            save_as_itk(pred_oct_vol.transpose(2, 0, 1),
                        os.path.join(nifti_result_path, f'{subject_name}_vol.nii.gz'), spacing)
            save_as_itk(pred_seg_vol.transpose(2, 0, 1),
                        os.path.join(nifti_result_path, f'{subject_name}_vol_seg.nii.gz'), spacing)


            # plot summary
            plot_final_summary_oct(pred_oct_vol, oct_vol, output_path, idx_smpl=str(subject_name))

            metric_results_path = os.path.join(output_path, 'metric_results')
            if not os.path.exists(metric_results_path):
                os.mkdir(metric_results_path)

            # Evaluate performance for reconstructed and interpolated B-Scans
            print('\nEvaluating reconstruction of OCT B-scans:')
            metrics = calculate_eval_metrics(oct_vol[..., idx_train], pred_oct_vol[..., idx_train], mode='2D')
            with open(os.path.join(metric_results_path, f'recon_results_OCT_smpl{subject_name}.pkl'), 'wb') as f:
                pickle.dump(metrics, f)

            print('Evaluating interpolated OCT B-scans:')
            metrics = calculate_eval_metrics(oct_vol[..., idx_interp], pred_oct_vol[..., idx_interp], mode='2D')
            with open(os.path.join(metric_results_path, 'interp_results_OCT_smpl{subject_name}.pkl'), 'wb') as f:
                pickle.dump(metrics, f)

            n = config.PREPROCESSING.BSCAN_SUBSAMPLE_FACTOR + 1
            n = 3
            interp_results_path = os.path.join(output_path, 'interpolation_results')
            if not os.path.exists(interp_results_path):
                os.mkdir(interp_results_path)

            for d in range(len(idx_train) - 1):
                fig, ax = plt.subplots(2, n, figsize=(4 * n, 6))
                for idx in range(n):
                    oct_slice = oct_vol[..., d * (n - 1) + idx]
                    ax[0, idx].imshow(oct_slice, cmap='gray')
                    ax[0, idx].title.set_text('GT slice ' + str(d * (n - 1) + idx))
                    ax[1, idx].imshow(pred_oct_vol[..., d * (n - 1) + idx], cmap='gray', vmin=oct_slice.min(),
                                      vmax=oct_slice.max())
                    if idx == 0 or idx == n - 1:
                        ax[1, idx].title.set_text('Recon. slice ' + str(d * (n - 1) + idx))
                    else:
                        ax[1, idx].title.set_text('Interp. slice ' + str(d * (n - 1) + idx))
                plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
                plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

                plt.savefig(os.path.join(interp_results_path, f'interpResult_{subject_name}_slice{d}.png'))
                plt.close()

        print("[EVAL] Mean over all samples - loss_recon: {:.6f} - loss_seg: {:.6f} ".format(np.array(recon_losses).mean(),
                                                                                             np.array(seg_losses).mean()))