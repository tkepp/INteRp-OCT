#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_step_summary_oct_with_seg(oct_slice_pred, oct_slice_gt, seg_slice_pred, seg_slice_gt, loss, h_init, h_curr, save_dir, step, max_label=12):

    layout = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"], ["J", "J", "J"]]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 10))

    img_oct_gt = axes['A'].imshow(oct_slice_gt, cmap='gray')
    axes['A'].set_title('OCT ground truth')
    img_oct_pred = axes['B'].imshow(oct_slice_pred, cmap='gray', vmin=oct_slice_gt.min(), vmax=oct_slice_gt.max())
    axes['B'].set_title('OCT prediction')
    img_diff = axes['C'].imshow(np.abs(oct_slice_pred-oct_slice_gt), cmap='hot', vmin=0, vmax=1)
    axes['C'].set_title(' absolute difference')
    #
    img_seg_gt = axes['D'].imshow(seg_slice_gt, cmap='gray')
    axes['D'].set_title('Segm. ground truth')
    img_seg_pred = axes['E'].imshow(seg_slice_pred, cmap='gray', vmin=seg_slice_gt.min(), vmax=seg_slice_gt.max())
    axes['E'].set_title('Segm. prediction')
    img_diff_seg = axes['F'].imshow(np.abs(seg_slice_pred-seg_slice_gt), cmap='hot', vmin=0, vmax=0.5)
    axes['F'].set_title(' absolute difference')
    #
    num_lc = h_init.shape[0]
    len_lc = h_init.shape[1]
    lc_ar = float(len_lc/num_lc)
    img_h_init = axes['G'].imshow(h_init, cmap='viridis', interpolation='none', aspect=lc_ar,
                                  vmin=h_init.min(), vmax=h_init.max())
    axes['G'].set_yticks(np.arange(num_lc))
    axes['G'].set_title('Init. latent code')
    img_h_curr = axes['H'].imshow(h_curr, cmap='viridis', interpolation='none', aspect=lc_ar)
    axes['H'].set_yticks(np.arange(num_lc))
    axes['H'].set_title('Curr. latent code')
    img_h_diff = axes['I'].imshow(np.abs(h_init - h_curr), cmap='hot', interpolation='none', aspect=lc_ar)
    axes['I'].set_yticks(np.arange(num_lc))
    axes['I'].set_title(' absolute difference')
    #
    axes['J'].plot(loss, 'b--', label='loss')
    axes['J'].set_title('loss')
    axes['J'].legend(loc='upper right')
    axes['J'].grid(True)
    axes['J'].set_xlabel("epochs")
    axes['J'].set_ylabel("loss")
    axes['J'].set_xlim(left=0)
    axes['J'].set_ylim(bottom=0, top=np.max(loss))
    #
    cbar1 = fig.colorbar(img_oct_gt, ax=axes['A'], orientation='horizontal')
    cbar2 = fig.colorbar(img_oct_pred, ax=axes['B'], orientation='horizontal')
    cbar3 = fig.colorbar(img_diff, ax=axes['C'], orientation='horizontal')
    cbar4 = fig.colorbar(img_seg_gt, ax=axes['D'], orientation='horizontal')
    cbar5 = fig.colorbar(img_seg_pred, ax=axes['E'], orientation='horizontal')
    cbar6 = fig.colorbar(img_diff_seg, ax=axes['F'], orientation='horizontal')
    cbar7 = fig.colorbar(img_h_init, ax=axes['G'], orientation='horizontal')
    cbar8 = fig.colorbar(img_h_curr, ax=axes['H'], orientation='horizontal')
    cbar9 = fig.colorbar(img_h_diff, ax=axes['I'], orientation='horizontal')
    fig.tight_layout()
    # plt.legend([pred_slo_plot, gt_slo_plot],['prediction', 'ground truth'])
    # fig.suptitle('training summary on step ' + str(step), fontsize=16)
    plt.savefig(os.path.join(save_dir, 'oct_inr_train_step_{:04d}.png'.format(step+1)), dpi=128, bbox_inches='tight')
    plt.close(fig)


def plot_step_summary_oct_with_seg_single(oct_slice_pred, oct_slice_gt, seg_slice_pred, seg_slice_gt, loss, save_dir, step, max_label=12):

    layout = [["A", "B", "C"], ["D", "E", "F"], ["G", "G", "G"]]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 10))

    img_oct_gt = axes['A'].imshow(oct_slice_gt, cmap='gray')
    axes['A'].set_title('OCT ground truth')
    img_oct_pred = axes['B'].imshow(oct_slice_pred, cmap='gray', vmin=oct_slice_gt.min(), vmax=oct_slice_gt.max())
    axes['B'].set_title('OCT prediction')
    img_diff = axes['C'].imshow(np.abs(oct_slice_pred-oct_slice_gt), cmap='hot', vmin=0, vmax=1)
    axes['C'].set_title(' absolute difference')
    #
    img_seg_gt = axes['D'].imshow(seg_slice_gt, cmap='gray')
    axes['D'].set_title('Segm. ground truth')
    img_seg_pred = axes['E'].imshow(seg_slice_pred, cmap='gray', vmin=seg_slice_gt.min(), vmax=seg_slice_gt.max())
    axes['E'].set_title('Segm. prediction')
    img_diff_seg = axes['F'].imshow(np.abs(seg_slice_pred-seg_slice_gt), cmap='hot', vmin=0, vmax=0.5)
    axes['F'].set_title(' absolute difference')
    #
    axes['G'].plot(loss, 'b--', label='loss')
    axes['G'].set_title('loss')
    axes['G'].legend(loc='upper right')
    axes['G'].grid(True)
    axes['G'].set_xlabel("epochs")
    axes['G'].set_ylabel("loss")
    axes['G'].set_xlim(left=0)
    axes['G'].set_ylim(bottom=0, top=np.max(loss))
    #
    cbar1 = fig.colorbar(img_oct_gt, ax=axes['A'], orientation='horizontal')
    cbar2 = fig.colorbar(img_oct_pred, ax=axes['B'], orientation='horizontal')
    cbar3 = fig.colorbar(img_diff, ax=axes['C'], orientation='horizontal')
    cbar4 = fig.colorbar(img_seg_gt, ax=axes['D'], orientation='horizontal')
    cbar5 = fig.colorbar(img_seg_pred, ax=axes['E'], orientation='horizontal')
    cbar6 = fig.colorbar(img_diff_seg, ax=axes['F'], orientation='horizontal')
    fig.tight_layout()
    # plt.legend([pred_slo_plot, gt_slo_plot],['prediction', 'ground truth'])
    # fig.suptitle('training summary on step ' + str(step), fontsize=16)
    plt.savefig(os.path.join(save_dir, 'oct_inr_train_step_{:04d}.png'.format(step+1)), dpi=128, bbox_inches='tight')
    plt.close(fig)


def plot_step_summary(oct_slice_pred, oct_slice_gt, slo_line_pred, slo_line_gt, loss, save_dir, step):

    layout = [["A", "B", "C"], ["D", "D", "D"], ["E", "E", "E"]]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 10))

    img_oct_gt = axes['A'].imshow(oct_slice_gt, cmap='gray')
    axes['A'].set_title('OCT ground truth')
    img_oct_pred = axes['B'].imshow(oct_slice_pred, cmap='gray', vmin=oct_slice_gt.min(), vmax=oct_slice_gt.max())
    axes['B'].set_title('OCT prediction')
    img_diff = axes['C'].imshow(np.abs(oct_slice_pred-oct_slice_gt), cmap='hot', vmin=0, vmax=1)
    axes['C'].set_title(' absolute difference')
    axes['D'].plot(slo_line_pred, 'r--', label='prediction')
    axes['D'].plot(slo_line_gt, 'k', label='ground truth')
    axes['D'].set_ylim(bottom=slo_line_gt.min() - 0.1, top=slo_line_gt.max() + 0.1)
    axes['D'].grid(True)
    axes['D'].set_xlim(left=0, right=512)
    axes['D'].legend(loc='upper right')
    axes['D'].set_title('SLO prediction')
    axes['D'].set_xlabel("a-scans")
    axes['D'].set_ylabel("intensities")
    axes['E'].plot(loss, 'b--', label='loss')
    axes['E'].set_title('loss')
    axes['E'].legend(loc='upper right')
    axes['E'].grid(True)
    axes['E'].set_xlabel("epochs")
    axes['E'].set_ylabel("loss")
    axes['E'].set_xlim(left=0)
    axes['E'].set_ylim(bottom=0, top=1.5)
    cbar1 = fig.colorbar(img_oct_gt, ax=axes['A'], orientation='horizontal')
    cbar2 = fig.colorbar(img_oct_pred, ax=axes['B'], orientation='horizontal')
    cbar3 = fig.colorbar(img_diff, ax=axes['C'], orientation='horizontal')
    fig.tight_layout()
    # plt.legend([pred_slo_plot, gt_slo_plot],['prediction', 'ground truth'])
    fig.suptitle('training summary on step ' + str(step), fontsize=16)
    plt.savefig(os.path.join(save_dir, 'train_step_{:04d}.png'.format(step)), dpi=128, bbox_inches='tight')
    plt.close(fig)


def plot_final_summary(oct_pred, oct_gt, slo_pred, slo_gt, save_dir):

    layout = [["A", "B", "C", "D"], ["E", "F", "G", "H"], ["I", "J", "K", "L"]]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 10))
    H, W, D = oct_pred.shape
    img_oct_gt_d25 = axes['A'].imshow(oct_gt[..., D // 4], cmap='gray')
    axes['A'].set_title('OCT ground truth (slice {:d})'.format(D//4))
    img_oct_gt_d50 = axes['B'].imshow(oct_gt[..., D // 2], cmap='gray')
    axes['B'].set_title('OCT ground truth (slice {:d})'.format(D//2))
    img_oct_gt_d75 = axes['C'].imshow(oct_gt[..., 3 * D // 4], cmap='gray')
    axes['C'].set_title('OCT ground truth (slice {:d})'.format(3 * D // 4))
    img_slo_gt = axes['D'].imshow(slo_gt, cmap='gray', extent=[0,512*0.011678138747811317,0,13*0.4982675015926361])
    axes['D'].set_title('SLO ground truth')
    img_oct_pred_d25 = axes['E'].imshow(oct_pred[..., D // 4], cmap='gray', vmin=oct_gt[..., D // 4].min(), vmax=oct_gt[..., D // 4].max())
    axes['E'].set_title('OCT prediction (slice {:d})'.format(D // 4))
    img_oct_pred_d50 = axes['F'].imshow(oct_pred[..., D // 2], cmap='gray', vmin=oct_gt[..., D // 2].min(), vmax=oct_gt[..., D // 2].max())
    axes['F'].set_title('OCT prediction (slice {:d})'.format(D // 2))
    img_oct_pred_d75 = axes['G'].imshow(oct_pred[..., 3 * D // 4], cmap='gray', vmin=oct_gt[..., 3 * D // 4].min(), vmax=oct_gt[..., 3 * D // 4].max())
    axes['G'].set_title('OCT prediction (slice {:d})'.format(3 * D // 4))
    img_slo_pred = axes['H'].imshow(slo_pred, cmap='gray', extent=[0,512*0.011678138747811317,0,13*0.4982675015926361])
    axes['H'].set_title('SLO prediction')

    diff_oct_d25 = axes['I'].imshow(np.abs(oct_pred[..., D // 4] - oct_gt[..., D // 4]), cmap='hot', vmin=0, vmax=1)
    axes['I'].set_title('absolute difference')
    diff_oct_d50 = axes['J'].imshow(np.abs(oct_pred[..., D // 2] - oct_gt[..., D // 2]), cmap='hot', vmin=0, vmax=1)
    axes['J'].set_title('absolute difference')
    diff_oct_d75 = axes['K'].imshow(np.abs(oct_pred[..., 3 * D // 4] - oct_gt[..., 3 * D // 4]), cmap='hot', vmin=0, vmax=1)
    axes['K'].set_title('absolute difference')
    diff_slo = axes['L'].imshow(np.abs(slo_pred - slo_gt), cmap='hot', vmin=0, vmax=3, extent=[0,512*0.011678138747811317,0,13*0.4982675015926361])
    axes['L'].set_title('absolute difference')
    cbar1 = fig.colorbar(img_oct_gt_d25, ax=axes['A'], orientation='horizontal')
    cbar2 = fig.colorbar(img_oct_gt_d50, ax=axes['B'], orientation='horizontal')
    cbar3 = fig.colorbar(img_oct_gt_d75, ax=axes['C'], orientation='horizontal')
    cbar4 = fig.colorbar(img_slo_gt, ax=axes['D'], orientation='horizontal')
    cbar5 = fig.colorbar(img_oct_pred_d25, ax=axes['E'], orientation='horizontal')
    cbar6 = fig.colorbar(img_oct_pred_d50, ax=axes['F'], orientation='horizontal')
    cbar7 = fig.colorbar(img_oct_pred_d75, ax=axes['G'], orientation='horizontal')
    cbar8 = fig.colorbar(img_slo_pred, ax=axes['H'], orientation='horizontal')
    cbar9 = fig.colorbar(diff_oct_d25, ax=axes['I'], orientation='horizontal')
    cbar10 = fig.colorbar(diff_oct_d50, ax=axes['J'], orientation='horizontal')
    cbar11 = fig.colorbar(diff_oct_d75, ax=axes['K'], orientation='horizontal')
    cbar12 = fig.colorbar(diff_slo, ax=axes['L'], orientation='horizontal')
    fig.tight_layout()
    # plt.legend([pred_slo_plot, gt_slo_plot],['prediction', 'ground truth'])
    fig.suptitle('total training summary', fontsize=16)
    plt.savefig(os.path.join(save_dir, 'train_final.png'), dpi=128, bbox_inches='tight')
    plt.close(fig)


def plot_final_summary_oct(oct_pred, oct_gt, save_dir, idx_smpl=1):

    layout = [["A", "B", "C"], ["D", "E", "F"], ["G", "H", "I"]]

    fig, axes = plt.subplot_mosaic(layout, figsize=(10, 10))
    H, W, D = oct_pred.shape
    img_oct_gt_d25 = axes['A'].imshow(oct_gt[..., D // 4], cmap='gray')
    axes['A'].set_title('OCT ground truth (slice {:d})'.format(D//4))
    img_oct_gt_d50 = axes['B'].imshow(oct_gt[..., D // 2], cmap='gray')
    axes['B'].set_title('OCT ground truth (slice {:d})'.format(D//2))
    img_oct_gt_d75 = axes['C'].imshow(oct_gt[..., 3 * D // 4], cmap='gray')
    axes['C'].set_title('OCT ground truth (slice {:d})'.format(3 * D // 4))

    img_oct_pred_d25 = axes['D'].imshow(oct_pred[..., D // 4], cmap='gray', vmin=oct_gt[..., D // 4].min(), vmax=oct_gt[..., D // 4].max())
    axes['D'].set_title('OCT prediction (slice {:d})'.format(D // 4))
    img_oct_pred_d50 = axes['E'].imshow(oct_pred[..., D // 2], cmap='gray', vmin=oct_gt[..., D // 2].min(), vmax=oct_gt[..., D // 2].max())
    axes['E'].set_title('OCT prediction (slice {:d})'.format(D // 2))
    img_oct_pred_d75 = axes['F'].imshow(oct_pred[..., 3 * D // 4], cmap='gray', vmin=oct_gt[..., 3 * D // 4].min(), vmax=oct_gt[..., 3 * D // 4].max())
    axes['F'].set_title('OCT prediction (slice {:d})'.format(3 * D // 4))

    diff_oct_d25 = axes['G'].imshow(np.abs(oct_pred[..., D // 4] - oct_gt[..., D // 4]), cmap='hot', vmin=0, vmax=1)
    axes['G'].set_title('absolute difference')
    diff_oct_d50 = axes['H'].imshow(np.abs(oct_pred[..., D // 2] - oct_gt[..., D // 2]), cmap='hot', vmin=0, vmax=1)
    axes['H'].set_title('absolute difference')
    diff_oct_d75 = axes['I'].imshow(np.abs(oct_pred[..., 3 * D // 4] - oct_gt[..., 3 * D // 4]), cmap='hot', vmin=0, vmax=1)
    axes['I'].set_title('absolute difference')

    cbar1 = fig.colorbar(img_oct_gt_d25, ax=axes['A'], orientation='horizontal')
    cbar2 = fig.colorbar(img_oct_gt_d50, ax=axes['B'], orientation='horizontal')
    cbar3 = fig.colorbar(img_oct_gt_d75, ax=axes['C'], orientation='horizontal')

    cbar5 = fig.colorbar(img_oct_pred_d25, ax=axes['D'], orientation='horizontal')
    cbar6 = fig.colorbar(img_oct_pred_d50, ax=axes['E'], orientation='horizontal')
    cbar7 = fig.colorbar(img_oct_pred_d75, ax=axes['F'], orientation='horizontal')

    cbar9 = fig.colorbar(diff_oct_d25, ax=axes['G'], orientation='horizontal')
    cbar10 = fig.colorbar(diff_oct_d50, ax=axes['H'], orientation='horizontal')
    cbar11 = fig.colorbar(diff_oct_d75, ax=axes['I'], orientation='horizontal')

    fig.tight_layout()
    # plt.legend([pred_slo_plot, gt_slo_plot],['prediction', 'ground truth'])
    fig.suptitle('total training summary', fontsize=16)
    if isinstance(idx_smpl, str):
        plt.savefig(os.path.join(save_dir, f'{idx_smpl}_train_final.png'), dpi=128, bbox_inches='tight')
    else:
        plt.savefig(os.path.join(save_dir, '{:3}_train_final.png'.format(idx_smpl)), dpi=128, bbox_inches='tight')
    plt.close(fig)