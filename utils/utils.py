# -*- coding: utf-8 -*-
# @Author  : zhangwei
# @Time    : 2023/6/25 PM 4:01
# @Func    : utils

import os
import ants
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pingouin as pg
import seaborn as sns
import sh
from matplotlib.pyplot import MultipleLocator
from scipy import stats
from scipy.io import savemat, loadmat
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import fdrcorrection
from glob import glob
from tqdm import tqdm


# 设置环境变量
# os.environ['OUTDATED_IGNORE'] = '1'

def set_environ():
	# FreeSurfer
	value = os.environ.get('FREESURFER_HOME')
	if value is None:
		os.environ['FREESURFER_HOME'] = '/usr/local/freesurfer'
		os.environ['SUBJECTS_DIR'] = '/usr/local/freesurfer/subjects'
		os.environ['PATH'] = '/usr/local/freesurfer/bin:' + os.environ['PATH']
	# FSL
	os.environ['PATH'] = '/usr/local/fsl/bin:' + '/usr/local/workbench/bin_linux64:' + os.environ['PATH']
	os.environ['FSLOUTPUTTYPE'] = 'NIFTI_GZ'


def compute_fcmap(seed, bold):
	n_vertex = bold.shape[0]
	fcmap = np.zeros((n_vertex))
	for i in range(n_vertex):
		r, _ = stats.pearsonr(bold[i, :], seed)
		fcmap[i] = r
	return fcmap


def save_mgh(data, file):
	if data.dtype == np.float64 or data.dtype == np.bool:
		data = data.astype('float32')
	img = nib.MGHImage(data, np.eye(4))
	nib.save(img, file)


def get_vol_concate(vol_files):
	print(len(vol_files), vol_files)
	vol_bold = np.zeros(shape=(128, 128, 128, 0))
	for bold_file in vol_files:
		tmp_bold = ants.image_read(bold_file)
		tmp_bold_vol = tmp_bold.numpy()
		vol_bold = np.concatenate((vol_bold, tmp_bold_vol), axis=3)
	print(vol_bold.shape)
	return vol_bold.astype(np.float16)


def get_surf_concate(surf_bold_files, fs_n_vertex):
	print(len(surf_bold_files), surf_bold_files)
	surf_bold = np.zeros(shape=(fs_n_vertex, 0))
	for bold_file in surf_bold_files:
		tmp_bold = nib.load(bold_file).get_fdata()
		n_frame = tmp_bold.shape[3]
		n_vertex = tmp_bold.shape[0] * tmp_bold.shape[1] * tmp_bold.shape[2]
		tmp_bold_surf = tmp_bold.reshape((n_vertex, n_frame), order='F')
		surf_bold = np.hstack((surf_bold, tmp_bold_surf))
	print(surf_bold.shape)
	return surf_bold.astype(np.float16)


def cortex_roi_fc_mean(fc_lh, fc_rh, mask_lh, mask_rh):
	"""
	Calculate the mean FC intensity at the mask locations, return the average results for both hemispheres.
	"""
	lh_intensity = np.mean(fc_lh[mask_lh != 0], axis=0)
	rh_intensity = np.mean(fc_rh[mask_rh != 0], axis=0)
	bilateral_intensity = (lh_intensity + rh_intensity) / 2
	return bilateral_intensity


def mri_surf2surf_nnf(srcsubject, sval, trgsubject, tval, hemi, *args):
	sh.mri_surf2surf(
		'--srcsubject', srcsubject,
		'--sval', sval,
		'--trgsubject', trgsubject,
		'--tval', tval,
		'--mapmethod', 'nnf',
		# '--nsmooth-out', 1,
		# '--nsmooth-in', 1,
		'--hemi', hemi, *args)


def mri_surf2surf(srcsubject, sval, trgsubject, tval, hemi, *args):
	sh.mri_surf2surf(
		'--srcsubject', srcsubject,
		'--sval', sval,
		'--trgsubject', trgsubject,
		'--tval', tval,
		'--mapmethod', 'nnfr',
		# '--nsmooth-out', 1,
		# '--nsmooth-in', 1,
		'--hemi', hemi, *args)


def mri_surf2surf_annot(srcsubject, sval, trgsubject, tval, hemi, *args):
	sh.mri_surf2surf(
		'--srcsubject', srcsubject,
		'--sval-annot', sval,
		'--trgsubject', trgsubject,
		'--tval', tval,
		'--hemi', hemi, *args)


def mri_surf2surf_smooth(srcsubject, sval, trgsubject, tval, hemi, smooth_N, *args):
	sh.mri_surf2surf(
		'--srcsubject', srcsubject,
		'--sval', sval,
		'--trgsubject', trgsubject,
		'--tval', tval,
		'--hemi', hemi,
		'--fwhm-trg', smooth_N,
		*args)


def mris_fwhm(input, subject, hemi, fwhm, output, *args):
	"""
	 mris_fwhm --i lh.parc1_0004.mgh --subject fsaverage6 --hemi lh --fwhm 6  --o lh.parc1_0004_sm.mgh
	"""
	sh.mris_fwhm(
		'--i', input,
		'--subject', subject,
		'--hemi', hemi,
		'--fwhm', fwhm,
		'--o', output, *args)


def save_mgh(data, file):
	if data.dtype == np.float64 or data.dtype == np.bool:
		data = data.astype('float32')
	img = nib.MGHImage(data, np.eye(4))
	nib.save(img, file)


def concate_surf_bold_files(surf_bold_files):
	fs6_n_vertex = 40962
	surf_bold = np.zeros(shape=(fs6_n_vertex, 0))
	for bold_file in surf_bold_files:
		tmp_bold = nib.load(bold_file).get_fdata()
		n_frame = tmp_bold.shape[3]
		n_vertex = tmp_bold.shape[0] * tmp_bold.shape[1] * tmp_bold.shape[2]
		tmp_bold_surf = tmp_bold.reshape((n_vertex, n_frame), order='F')
		surf_bold = np.hstack((surf_bold, tmp_bold_surf))
	return surf_bold


def get_annot_parc_info(parc_files):
	parc_labels, _, names = nib.freesurfer.read_annot(str(parc_files))
	parc_labels = parc_labels.astype(np.int)
	parc_names = [name.decode() for name in names]
	print(len(parc_labels), parc_labels)
	print(len(names), names)
	return parc_labels, parc_names


def compute_fcmap(seed, bold):
	n_vertex = bold.shape[0]
	fcmap = np.zeros((n_vertex))
	for i in range(n_vertex):
		r, _ = stats.pearsonr(bold[i, :], seed)
		fcmap[i] = r
	return fcmap


def montage(lh_0, lh_180, rh_0, rh_180, pic_file):
	img_lh_pial_0 = cv2.imread(lh_0)
	img_lh_pial_180 = cv2.imread(lh_180)
	img_rh_pial_0 = cv2.imread(rh_0)
	img_rh_pial_180 = cv2.imread(rh_180)
	img_pial_up = np.hstack((img_lh_pial_0[150:420, 120:470, :], img_rh_pial_0[150:420, 120:470, :]))
	img_pial_down = np.hstack(
		(img_lh_pial_180[150:420, 120:470, :], img_rh_pial_180[150:420, 120:470, :]))
	img_pial = np.vstack((img_pial_up, img_pial_down))
	cv2.imwrite(pic_file, img_pial)


def montage_sym(lh_0, lh_180, rh_0, rh_180, pic_file):
	img_lh_pial_0 = cv2.imread(lh_0)
	img_lh_pial_180 = cv2.imread(lh_180)
	img_rh_pial_0 = cv2.imread(rh_0)
	img_rh_pial_180 = cv2.imread(rh_180)
	img_pial_up = np.hstack((img_lh_pial_0[120:450, 80:520, :], img_rh_pial_0[120:450, 80:520, :]))
	img_pial_down = np.hstack(
		(img_lh_pial_180[120:450, 80:520, :], img_rh_pial_180[120:450, 80:520, :]))
	img_pial = np.vstack((img_pial_up, img_pial_down))
	cv2.imwrite(pic_file, img_pial)


def regress_plot(xx, yy, x_min, x_max, y_min, y_max, MultipleLocator_x, MultipleLocator_y, work_dir, png_name):
	# Regression plot
	# plt.scatter(feature_mean, label, c='b', s=50)
	plt.figure(figsize=(5, 5))
	# plt.figure(figsize=(9, 5))
	# sns.regplot(x=xx, y=yy, color='black').invert_yaxis()
	sns.regplot(x=xx, y=yy, color='black', fit_reg=True, ci=95)
	# bwith = 3  # 边框宽度设置
	# tick_length = 10
	bwith = 2.5  # 边框宽度设置
	tick_length = 8
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	# plt.rc('font', family='Arial')
	plt.rcParams['font.sans-serif'] = ['Arial']
	# x_min, x_max = 0, 1
	# y_min, y_max = 0, 20
	plt.ylim(y_min, y_max)
	plt.xlim(x_min, x_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	x_major_locator = MultipleLocator(MultipleLocator_x)
	TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=22, color='#000000')
	plt.xticks(fontsize=22, color='#000000')
	plt.tick_params(width=bwith, length=tick_length, labelsize=22)
	plt.xlabel(" ")
	plt.ylabel(" ")
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(work_dir, f'scatter_{png_name}.png'), dpi=200)
	# plt.show()
	plt.close()


def regress_plot_color(xx, yy, x_min, x_max, y_min, y_max, MultipleLocator_x, MultipleLocator_y, work_dir, png_name):
	# Regression plot
	# plt.scatter(feature_mean, label, c='b', s=50)
	plt.figure(figsize=(5, 5))
	# plt.figure(figsize=(9, 5))
	# sns.regplot(x=xx, y=yy, color='black').invert_yaxis()
	# sns.regplot(x=xx, y=yy, color='black', fit_reg=True, ci=95)
	sns.regplot(x=xx, y=yy, color='black', fit_reg=True, ci=95, scatter=False)
	# 使用 seaborn.regplot 绘制回归图，根据 Category 划分颜色
	sns.scatterplot(x=xx, y=yy, hue=yy)

	# bwith = 3  # 边框宽度设置
	# tick_length = 10
	bwith = 2.5  # 边框宽度设置
	tick_length = 8
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	# plt.rc('font', family='Arial')
	plt.rcParams['font.sans-serif'] = ['Arial']
	# x_min, x_max = 0, 1
	# y_min, y_max = 0, 20
	plt.ylim(y_min, y_max)
	plt.xlim(x_min, x_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	x_major_locator = MultipleLocator(MultipleLocator_x)
	TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=22, color='#000000')
	plt.xticks(fontsize=22, color='#000000')
	plt.tick_params(width=bwith, length=tick_length, labelsize=22)
	plt.xlabel(" ")
	plt.ylabel(" ")
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(work_dir, f'scatter_{png_name}.png'), dpi=200)
	# plt.show()
	plt.close()


def regress_plot_color_custom(xx, yy, x_min, x_max, y_min, y_max, MultipleLocator_x, MultipleLocator_y, work_dir,
                              png_name):
	# Regression plot
	# plt.scatter(feature_mean, label, c='b', s=50)
	plt.figure(figsize=(5, 5))
	# plt.figure(figsize=(9, 5))
	# sns.regplot(x=xx, y=yy, color='black').invert_yaxis()
	# sns.regplot(x=xx, y=yy, color='black', fit_reg=True, ci=95)
	sns.regplot(x=xx, y=yy, color='black', fit_reg=True, ci=95, scatter=False)
	# 使用 seaborn.regplot 绘制回归图，根据 Category 划分颜色
	sns.scatterplot(x=xx[:10], y=yy[:10], color=['orange'])
	sns.scatterplot(x=xx[10:], y=yy[10:], color=['green'])

	# bwith = 3  # 边框宽度设置
	# tick_length = 10
	bwith = 2.5  # 边框宽度设置
	tick_length = 8
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	# plt.rc('font', family='Arial')
	plt.rcParams['font.sans-serif'] = ['Arial']
	# x_min, x_max = 0, 1
	# y_min, y_max = 0, 20
	plt.ylim(y_min, y_max)
	plt.xlim(x_min, x_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	x_major_locator = MultipleLocator(MultipleLocator_x)
	TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=22, color='#000000')
	plt.xticks(fontsize=22, color='#000000')
	plt.tick_params(width=bwith, length=tick_length, labelsize=22)
	plt.xlabel(" ")
	plt.ylabel(" ")
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(work_dir, f'scatter_{png_name}.png'), dpi=200)
	# plt.show()
	plt.close()


def volin_plot(data, x, y, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 8  # 字体大小

	# sns.boxplot(x=x, y=y, data=data)

	# sns.barplot(x=x, y=y, data=data)
	# 在提琴图上添加每个数据点
	sns.stripplot(x=x, y=y, data=data, jitter=True, color='black', size=3,
	              alpha=0.6)
	sns.violinplot(x=x, y=y, data=data)
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-60)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'volin_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def volin_plot_hue(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 8  # 字体大小

	# sns.violinplot(x=x, y=y, data=data)
	sns.boxplot(x=x, y=y, data=data, hue=hue)

	# sns.barplot(x=x, y=y, data=data)
	# 在提琴图上添加每个数据点
	sns.stripplot(x=x, y=y, data=data, jitter=True, color='black', size=3,
	              alpha=0.6)
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-60)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'volin_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def box_plot(data, x, y, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	data = data.sort_values(by=y, ascending=False)
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 8  # 字体大小
	sns.boxplot(data=data, x=x, y=y)
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-60)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'strip_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot(data, x, y, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name, order_list=None):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 10  # 字体大小
	sns.barplot(data=data, x=x, y=y, edgecolor='black', errcolor="black", errwidth=bwith, order=order_list)
	# sns.barplot(data=data, x=x, y=y, errorbar=("se"), edgecolor='black', errcolor="black", errwidth=bwith,  order=['Control', 'Preop', '1-month', '3-month', '6-month', '12-month'])
	# sns.barplot(data=data, x=x, y=y, edgecolor='black', errcolor="black", errwidth=bwith, order=['SCAN', 'Cing-Operc', 'Salience', 'Foot-SM', 'Hand-SM', 'Face-SM'])
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])
	# plt.legend(loc="upper right", frameon=False)

	# 画出 y=0.25 这条水平线
	plt.axhline(-2, c="r", ls="--", lw=2)

	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-90)
	# plt.xticks(fontsize=fontsize, color='#000000', rotation=0)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title(png_name)
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_strip_plot(data, df, x, y, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])
	ax = sns.barplot(data=data, x=x, y=y, edgecolor='black', errcolor="black", errwidth=bwith, alpha=0.9)
	sns.stripplot(data=data, x=x, y=y, edgecolor='black', color='grey', jitter=False)
	for idx in df.index:
		# ax.plot([1, 2], df.loc[idx, ['Pre', 'Post']],
		ax.plot([0, 1], df.loc[idx, ['OFF', 'ON']],
		        # ax.plot([0, 1, 2, 3], df.loc[idx, ['1m', '3m', '6m', '12m']],
		        # ax.plot([0, 3], df.loc[idx, ['1m', '12m']],
		        color='black', linewidth=0.5, linestyle='--', zorder=-1)
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000')
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_strip_plot_debug(data, df, x, y, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])
	ax = sns.barplot(data=data, x=x, y=y, edgecolor='black', errcolor="black", errwidth=bwith, alpha=0.9)
	sns.stripplot(data=data, x=x, y=y, edgecolor='black', color='grey', jitter=False)
	for idx in df.index:
		ax.plot([0, 1], df.loc[idx, ['Pre', 'Post']],
		        # ax.plot([0, 1], df.loc[idx, ['Pre', 'ON']],
		        # ax.plot([0, 1, 2, 3], df.loc[idx, ['1m', '3m', '6m', '12m']],
		        # ax.plot([0, 3], df.loc[idx, ['1m', '12m']],
		        color='black', linewidth=0.5, linestyle='--', zorder=-1)
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000')
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_horizontal(data, x, y, x_min, x_max, figsize, MultipleLocator_x, save_path, png_name):
	data = data.sort_values(by=x, ascending=False)
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	sns.barplot(data=data, x=x, y=y, edgecolor='black', errcolor="black", errwidth=bwith)
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])
	# plt.legend(loc="upper right", frameon=False)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.xlim(x_min, x_max)
	x_major_locator = MultipleLocator(MultipleLocator_x)
	TK.xaxis.set_major_locator(x_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000')
	plt.tick_params(bottom=True, left=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group_with_stats(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	# 计算t-test结果并添加标记
	groups = data[hue].unique()
	rois = data[x].unique()
	for i in range(len(rois)):
		group1 = data[(data[hue] == groups[0]) & (data[x] == rois[i])][y].values
		group2 = data[(data[hue] == groups[1]) & (data[x] == rois[i])][y].values
		t_stat, p_value = calculate_ttest_indipendent(group1, group2)
		print(f'{png_name},roi: {rois[i]}, t:{t_stat}, p:{p_value}')
		if p_value < 0.001:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '***', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.01:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '**', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.05:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '*', ha='center', va='bottom', color='#FF0000')
			continue
		if 0.05 < p_value < 0.1:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '+', ha='center', va='bottom', color='#FF0000')

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	# plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-90)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title(png_name)
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group_with_stats_MannWhitneyU(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path,
                                           png_name):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	# 计算mannwhitneyU结果并添加标记
	groups = data[hue].unique()
	rois = data[x].unique()
	for i in range(len(rois)):
		group1 = data[(data[hue] == groups[0]) & (data[x] == rois[i])][y].values
		group2 = data[(data[hue] == groups[1]) & (data[x] == rois[i])][y].values
		t_stat, p_value = mannwhitneyu(group1, group2)
		if p_value < 0.001:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '***', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.01:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '**', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.05:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '*', ha='center', va='bottom', color='#FF0000')
			continue
		if 0.05 < p_value < 0.1:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '+', ha='center', va='bottom', color='#FF0000')

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group_with_order(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name,
                              x_order=None, hue_order=None):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	ax = sns.barplot(data=data, x=x, y=y, order=x_order, hue=hue, hue_order=hue_order, edgecolor='black',
	                 errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group_with_order_with_stats(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y, save_path, png_name,
                                         x_order=None, hue_order=None):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	# ax = sns.barplot(data=data, x=x, y=y, order=x_order, hue=hue, hue_order=hue_order, edgecolor='black', errcolor="black", errwidth=bwith, errorbar=("se"))
	ax = sns.barplot(data=data, x=x, y=y, order=x_order, hue=hue, hue_order=hue_order, edgecolor='black',
	                 errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, order=x_order, hue=hue, hue_order=hue_order, edgecolor='black', errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	# 计算t-test结果并添加标记
	groups = data[hue].unique()
	rois = x_order
	for i in range(len(rois)):
		group1 = data[(data[hue] == groups[0]) & (data[x] == rois[i])][y].values
		group2 = data[(data[hue] == groups[1]) & (data[x] == rois[i])][y].values
		t_stat, p_value = calculate_ttest_indipendent(group1, group2)
		print('=====!!!!', rois[i], t_stat, p_value)
		if p_value < 0.001:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '***', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.01:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '**', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.05:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '*', ha='center', va='bottom', color='#FF0000')
			continue
		if 0.05 < p_value < 0.1:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '+', ha='center', va='bottom', color='#FF0000')

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	# plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.xticks(fontsize=fontsize, color='#000000', rotation=0)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def bar_plot_group_with_order_with_stats_MannWhitneyU(data, x, y, hue, y_min, y_max, figsize, MultipleLocator_y,
                                                      save_path, png_name, x_order=None, hue_order=None):
	plt.figure(figsize=figsize)
	bwith = 1.5  # 边框宽度设置
	tick_length = 3  # tick长度
	fontsize = 12  # 字体大小
	ax = sns.barplot(data=data, x=x, y=y, order=x_order, hue=hue, hue_order=hue_order, edgecolor='black',
	                 errcolor="black", errwidth=bwith)
	# ax = sns.barplot(data=data, x=x, y=y, hue=hue, edgecolor='black', errcolor="black", errwidth=bwith, hue_order=['Pre', 'OFF', 'ON', 'Control'])
	# ax.set_xticks([])  # 去除坐标轴刻度
	# sns.barplot(data=data, x=x, y=y, order=['DBSControl', 'DBSPatient', 'CH_1m', 'CH_3m', 'CH_6m', 'CH_12m'])

	# 画出 y=0.25 这条水平线
	# plt.axhline(0.25, c="r", ls="--", lw=2)

	# 计算t-test结果并添加标记
	groups = data[hue].unique()
	rois = x_order
	for i in range(len(rois)):
		group1 = data[(data[hue] == groups[0]) & (data[x] == rois[i])][y].values
		group2 = data[(data[hue] == groups[1]) & (data[x] == rois[i])][y].values
		t_stat, p_value = mannwhitneyu(group1, group2)
		if p_value < 0.001:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '***', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.01:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '**', ha='center', va='bottom', color='#FF0000')
			continue
		if p_value < 0.05:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '*', ha='center', va='bottom', color='#FF0000')
			continue
		if 0.05 < p_value < 0.1:
			x_position = i  # 设置标记的x位置在两组之间的中心
			plt.text(x_position, max(group1.mean(), group2.mean()), '+', ha='center', va='bottom', color='#FF0000')

	plt.legend(loc="upper right", frameon=False, fontsize=8)
	TK = plt.gca()  # 获取边框
	TK.spines['bottom'].set_linewidth(bwith)  # 图框下边
	TK.spines['left'].set_linewidth(bwith)  # 图框左边
	plt.rc('font', family='Arial')
	plt.ylim(y_min, y_max)
	y_major_locator = MultipleLocator(MultipleLocator_y)
	TK.yaxis.set_major_locator(y_major_locator)
	# x_major_locator = MultipleLocator(MultipleLocator_x)
	# TK.xaxis.set_major_locator(x_major_locator)
	plt.yticks(fontsize=fontsize, color='#000000')
	plt.xticks(fontsize=fontsize, color='#000000', rotation=-30)
	plt.tick_params(bottom=False, width=bwith, length=tick_length, labelsize=fontsize)  # bar图底部不要tick
	plt.xlabel(" ")
	plt.ylabel(" ")
	# TK.axes.xaxis.set_visible(False)  # 设置不显示横坐标label
	# plt.title('feature 44 fc intensity(sum) histogram')
	sns.despine()
	plt.tight_layout()

	plt.savefig(os.path.join(save_path, f'bar_{png_name}.png'), bbox_inches='tight', dpi=200)
	# plt.show()
	plt.close()


def calculate_corr_pearson(vec1, vec2):
	r, p = stats.pearsonr(vec1, vec2)
	return r, p


def calculate_corr_spearman(vec1, vec2):
	r, p = stats.spearmanr(vec1, vec2)
	return r, p


def calculate_partial_corr_pearson(vec1, vec2, vec_covar):
	covar_num = len(vec_covar)
	data = {'vec1': vec1,
	        'vec2': vec2}
	df = pd.DataFrame(data, columns=['vec1', 'vec2'])
	for i in range(covar_num):
		df[f"vec_covar{i}"] = vec_covar[i]
	# print(df)
	if covar_num == 1:
		pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar1'], method="pearson")
	if covar_num == 2:
		pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar1', 'vec_covar2'], method="pearson")
	print("partial_corr:")
	print(pc)
	r, p = pc['r'][0], pc['p-val'][0]
	return r, p


def calculate_partial_corr_spearman(vec1, vec2, vec_covar):
	covar_num = len(vec_covar)
	data = {'vec1': vec1,
	        'vec2': vec2}
	df = pd.DataFrame(data, columns=['vec1', 'vec2'])
	for i in range(covar_num):
		df[f"vec_covar{i}"] = vec_covar[i]
	# print(df)
	if covar_num == 1:
		pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0'], method="spearman")
	if covar_num == 2:
		pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0', 'vec_covar1'], method="spearman")
	print("partial_corr:")
	print(pc)
	r, p = pc['r'][0], pc['p-val'][0]
	return r, p


def calculate_partial_corr_custom(vec1, vec2, vec_covar, method='pearson'):
	"""
	从vec1中回归vec_covar,计算回归后的vec1 和 vec2 相关性
	"""
	Y = vec1
	X = np.asarray(vec_covar).reshape(-1, 1)
	reg = LinearRegression().fit(X, Y)
	Y_predicted = reg.predict(X)
	vec1_residuals = Y - Y_predicted

	Y = vec2
	X = np.asarray(vec_covar).reshape(-1, 1)
	reg = LinearRegression().fit(X, Y)
	Y_predicted = reg.predict(X)
	vec2_residuals = Y - Y_predicted

	if method == 'spearman':
		r, p = calculate_corr_spearman(vec1_residuals, vec2_residuals)
	if method == 'pearson':
		r, p = calculate_corr_pearson(vec1_residuals, vec2_residuals)
	print('custom correlation', r, p)

	# ==== check结果是否和partial_corr计算出的结果一致
	covar_num = len(vec_covar)
	data = {'vec1': vec1,
	        'vec2': vec2}
	df = pd.DataFrame(data, columns=['vec1', 'vec2'])
	for i in range(covar_num):
		df[f"vec_covar{i}"] = vec_covar[i]
	if method == 'spearman':
		if covar_num == 1:
			pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0'], method="spearman")
		if covar_num == 2:
			pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0', 'vec_covar1'], method="spearman")
	if method == 'pearson':
		if covar_num == 1:
			pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0'], method="pearson")
		if covar_num == 2:
			pc = pg.partial_corr(data=df, x='vec1', y='vec2', covar=['vec_covar0', 'vec_covar1'], method="pearson")
	r, p = pc['r'][0], pc['p-val'][0]
	print("partial_corr:")
	print(pc)

	return r, p, vec1_residuals, vec2_residuals


def calculate_ttest_indipendent(vec1, vec2):
	t, p = stats.ttest_ind(vec1, vec2)
	return t, p


def calculate_ttest_paired(vec1, vec2):
	t, p = stats.ttest_rel(vec1, vec2)
	return t, p


def calculate_ranksums_twotial(vec1, vec2):
	t, p = stats.ranksums(vec1, vec2)
	return t, p


def corr_heatmap(data, min, max, corr_method):
	# 可视化特征之间的相关性
	corr = data.corr(method=corr_method)
	corr.style.background_gradient(cmap='coolwarm').set_precision(2)
	# sns.heatmap(corr, vmin=min, vmax=max, cmap="coolwarm", annot=True)
	sns.heatmap(corr, vmin=min, vmax=max, cmap="coolwarm", annot=False)
	plt.show()
	return corr


def corr_heatma_v2(data, min_val, max_val, corr_method, alpha=0.05):
	# 可视化特征之间的相关性
	corr = data.corr(method=corr_method)
	sns.heatmap(corr, vmin=min_val, vmax=max_val, cmap="coolwarm", annot=True)

	# 添加相关性系数的p值是否显著的标记
	for i in range(len(corr)):
		for j in range(len(corr)):
			if i != j:
				if corr_method == 'pearson':
					corr_val, p_val = calculate_corr_pearson(data.iloc[:, i], data.iloc[:, j])
					if p_val < alpha:
						plt.text(j + 0.5, i + 0.3, "*", ha='center', va='center', color='black', fontsize=10)
					else:
						plt.text(j + 0.5, i + 0.3, "ns", ha='center', va='center', color='black', fontsize=10)
				if corr_method == 'spearman':
					corr_val, p_val = calculate_corr_spearman(data.iloc[:, i], data.iloc[:, j])
					if p_val < alpha:
						plt.text(j + 0.5, i + 0.3, "*", ha='center', va='center', color='black', fontsize=10)
					else:
						plt.text(j + 0.5, i + 0.3, "ns", ha='center', va='center', color='black', fontsize=10)

	plt.show()
	return corr


def read_txt(txt_file):
	with open(txt_file, "r", encoding='utf-8') as f:  # 打开文本
		data = f.read()  # 读取文本
	return data


def read_txt_line(txt_file):
	with open(txt_file, "r", encoding='utf-8') as f:  # 打开文本
		data = f.readlines()  # 读取文本
	data_clear = [i.strip('\n') for i in data]
	return data_clear


def load_surface_roi(roi_file):
	roi = nib.freesurfer.read_morph_data(roi_file) == 1
	return roi


def FDR_correction(p_list):
	# FDR correction
	_, FDR_p = fdrcorrection(np.array(p_list), method='i')
	return FDR_p


def save_array_to_mat(arr, name, file_path):
	"""
	将数组保存为MATLAB格式的.mat文件

	参数：
	- arr: 要保存的数组
	- name: 在MATLAB中使用的变量名
	- file_path: 保存的文件路径
	"""
	data_dict = {name: arr}
	savemat(file_path, data_dict)
	print(f'Data saved to {file_path}')


def load_mat_to_array(file_path, variable_name):
	"""
	从MATLAB格式的.mat文件中加载数据到数组

	参数：
	- file_path: .mat文件的路径
	- variable_name: 在MATLAB文件中的变量名

	返回：
	- 加载的数组
	"""
	mat_data = loadmat(file_path)
	return mat_data[variable_name]


def save_array_to_mat_v2(arr1, name1, arr2, name2, file_path):
	"""
	Save arrays to a MATLAB format .mat file.

	Parameters:
	- arr1: The first array to save.
	- name1: The variable name for the first array in MATLAB.
	- arr2: The second array to save.
	- name2: The variable name for the second array in MATLAB.
	- file_path: The file path to save the .mat file.
	"""
	data_dict = {name1: arr1, name2: arr2}
	savemat(file_path, data_dict)
	print(f'Data saved to {file_path}')


def load_mat_to_array_v2(file_path, variable_name1, variable_name2):
	"""
	Load data from a MATLAB format .mat file into arrays.

	Parameters:
	- file_path: The file path of the .mat file.
	- variable_name1: The variable name of the first array in the MATLAB file.
	- variable_name2: The variable name of the second array in the MATLAB file.

	Returns:
	- The loaded arrays.
	"""
	mat_data = loadmat(file_path)
	return mat_data[variable_name1], mat_data[variable_name2]


def compute_centroid(matrix):
	"""
	计算3D矩阵的几何中心。

	参数:
	matrix (numpy.ndarray): 一个3D矩阵。

	返回:
	tuple: 包含x, y, z坐标的元组。
	"""
	x_indices, y_indices, z_indices = np.where(matrix > 0)

	if len(x_indices) == 0:
		return (np.nan, np.nan, np.nan)

	x_center = np.mean(x_indices)
	y_center = np.mean(y_indices)
	z_center = np.mean(z_indices)

	return (x_center, y_center, z_center)


if __name__ == '__main__':
	pass
