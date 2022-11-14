# Copyright © 2016-2021 Medical Image Analysis Laboratory, University Hospital Center and University of Lausanne (UNIL-CHUV), Switzerland
#
#  This software is distributed under the open-source license Modified BSD.

"""PyMIALSRTK preprocessing functions.

It includes BTK Non-local-mean denoising, slice intensity correction
slice N4 bias field correction, slice-by-slice correct bias field, intensity standardization,
histogram normalization and both manual or deep learning based automatic brain extraction.

"""

import os
import traceback
from glob import glob
import pathlib

from skimage.morphology import binary_opening, binary_closing

import numpy as np
from traits.api import *

import nibabel

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import skimage.measure
from scipy.signal import argrelextrema
import scipy.ndimage as snd
import pandas as pd
import cv2

from nipype.utils.filemanip import split_filename
from nipype.interfaces.base import traits, \
    TraitedSpec, File, InputMultiPath, OutputMultiPath, BaseInterface, BaseInterfaceInputSpec

from pymialsrtk.interfaces.utils import run


####################
# Brain Extraction
####################


class BrainExtractionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    bids_dir = Directory(desc='Root directory', mandatory=True, exists=True)
    in_file = File(desc='Input image', mandatory=True)
    in_ckpt_loc = File(desc='Network_checkpoint for localization', mandatory=True)
    threshold_loc = traits.Float(0.49, desc='Threshold determining cutoff probability (0.49 by default)')
    in_ckpt_seg = File(desc='Network_checkpoint for segmentation', mandatory=True)
    threshold_seg = traits.Float(0.5, desc='Threshold for cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_brainMask", desc='Suffix of the automatically generated mask', usedefault=True)


class BrainExtractionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the BrainExtraction interface."""

    out_file = File(desc='Output brain mask image')


class BrainExtraction(BaseInterface):
    """Runs the automatic brain extraction module.

    This module is based on a 2D U-Net (Ronneberger et al. [1]_) using the pre-trained weights from Salehi et al. [2]_.

    References
    ------------
    .. [1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_
    .. [2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

    Examples
    --------
    >>> from pymialsrtk.interfaces.preprocess import BrainExtraction
    >>> brainMask = BrainExtraction()
    >>> brainmask.inputs.base_dir = '/my_directory'
    >>> brainmask.inputs.in_file = 'sub-01_acq-haste_run-1_2w.nii.gz'
    >>> brainmask.inputs.in_ckpt_loc = 'my_loc_checkpoint'
    >>> brainmask.inputs.threshold_loc = 0.49
    >>> brainmask.inputs.in_ckpt_seg = 'my_seg_checkpoint'
    >>> brainmask.inputs.threshold_seg = 0.5
    >>> brainmask.inputs.out_postfix = '_brainMask.nii.gz'
    >>> brainmask.run() # doctest: +SKIP

    """

    input_spec = BrainExtractionInputSpec
    output_spec = BrainExtractionOutputSpec

    def _gen_filename(self, name):
        if name == 'out_file':
            _, name, ext = split_filename(self.inputs.in_file)
            output = name + self.inputs.out_postfix + ext
            return os.path.abspath(output)
        return None

    def _run_interface(self, runtime):

        try:
            self._extractBrain(self.inputs.in_file, self.inputs.in_ckpt_loc, self.inputs.threshold_loc,
                               self.inputs.in_ckpt_seg, self.inputs.threshold_seg) #, self.inputs.bids_dir, self.inputs.out_postfix)
        except Exception:
            print('Failed')
            print(traceback.format_exc())
        return runtime

    def _extractBrain(self, dataPath, modelCkptLoc, thresholdLoc, modelCkptSeg, thresholdSeg): #, bidsDir, out_postfix):
        """Generate a brain mask by passing the input image(s) through two networks.

        The first network localizes the brain by a coarse-grained segmentation while the
        second one segments it more precisely. The function saves the output mask in the
        specific module folder created in bidsDir

        Parameters
        ----------
        dataPath <string>
            Input image file (required)

        modelCkptLoc <string>
            Network_checkpoint for localization (required)

        thresholdLoc <Float>
             Threshold determining cutoff probability (default is 0.49)

        modelCkptSeg <string>
            Network_checkpoint for segmentation

        thresholdSeg <Float>
             Threshold determining cutoff probability (default is 0.5)

        bidsDir <string>
            BIDS root directory (required)

        out_postfix <string>
            Suffix of the automatically generated mask (default is '_brainMask.nii.gz')

        """
        try:
            import tflearn  # noqa: E402
            from tflearn.layers.conv import conv_2d, max_pool_2d, upsample_2d  # noqa: E402
        except ImportError:
            print("tflearn not available. Can not run brain extraction")
            raise ImportError

        try:
            import tensorflow.compat.v1 as tf  # noqa: E402
        except ImportError:
            print("Tensorflow not available. Can not run brain extraction")
            raise ImportError

        # Step 1: Brain localization
        normalize = "local_max"
        width = 128
        height = 128
        border_x = 15
        border_y = 15
        n_channels = 1

        img_nib = nibabel.load(os.path.join(dataPath))
        image_data = img_nib.get_data()
        max_val = np.max(image_data)
        images = np.zeros((image_data.shape[2], width, height, n_channels))
        pred3dFinal = np.zeros((image_data.shape[2], image_data.shape[0], image_data.shape[1], n_channels))

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(image_data[:, :, ii],
                                  dsize=(width, height),
                                  fx=width, fy=height)
            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
                elif normalize == "global_max":
                    images[slice_counter, :, :, 0] = img_patch / max_val
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (img_patch - np.mean(img_patch)) / np.std(img_patch)
                else:
                    raise ValueError('Please select a valid normalization')
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope('inputs'):
                x = tf.placeholder(tf.float32, [None, width, height, n_channels], name='image')

            conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
            pool1 = max_pool_2d(conv1, 2)

            conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
            pool2 = max_pool_2d(conv2, 2)

            conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
            pool3 = max_pool_2d(conv3, 2)

            conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
            pool4 = max_pool_2d(conv4, 2)

            conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
            conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat', axis=3)
            conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge([up7, conv3], 'concat', axis=3)
            conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge([up8, conv2], 'concat', axis=3)
            conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge([up9, conv1], 'concat', axis=3)
            conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

            pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

        # Thresholding parameter to binarize predictions
        percentileLoc = thresholdLoc*100

        pred3d = []
        with tf.Session(graph=g) as sess_test_loc:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_loc, modelCkptLoc)

            for idx in range(images.shape[0]):

                im = np.reshape(images[idx, :, :, :], [1, width, height, n_channels])

                feed_dict = {x: im}
                pred_ = sess_test_loc.run(pred, feed_dict=feed_dict)

                theta = np.percentile(pred_, percentileLoc)
                pred_bin = np.where(pred_ > theta, 1, 0)
                pred3d.append(pred_bin[0, :, :, 0].astype('float64'))

            pred3d = np.asarray(pred3d)
            heights = []
            widths = []
            coms_x = []
            coms_y = []

            # Apply PPP
            ppp = True
            if ppp:
                pred3d = self._post_processing(pred3d)

            pred3d = [cv2.resize(elem,dsize=(image_data.shape[1], image_data.shape[0]), interpolation=cv2.INTER_NEAREST) for elem in pred3d]
            pred3d = np.asarray(pred3d)
            for i in range(np.asarray(pred3d).shape[0]):
                if np.sum(pred3d[i, :, :]) != 0:
                    pred3d[i, :, :] = self._extractLargestCC(pred3d[i, :, :].astype('uint8'))
                    contours, _ = cv2.findContours(pred3d[i, :, :].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    area = cv2.minAreaRect(np.squeeze(contours))
                    heights.append(area[1][0])
                    widths.append(area[1][1])
                    bbox = cv2.boxPoints(area).astype('int')
                    coms_x.append(int((np.max(bbox[:, 1])+np.min(bbox[:, 1]))/2))
                    coms_y.append(int((np.max(bbox[:, 0])+np.min(bbox[:, 0]))/2))
            # Saving localization points
            med_x = int(np.median(coms_x))
            med_y = int(np.median(coms_y))
            half_max_x = int(np.max(heights)/2)
            half_max_y = int(np.max(widths)/2)
            x_beg = med_x-half_max_x-border_x
            x_end = med_x+half_max_x+border_x
            y_beg = med_y-half_max_y-border_y
            y_end = med_y+half_max_y+border_y

        # Step 2: Brain segmentation
        width = 96
        height = 96

        images = np.zeros((image_data.shape[2], width, height, n_channels))

        slice_counter = 0
        for ii in range(image_data.shape[2]):
            img_patch = cv2.resize(image_data[x_beg:x_end, y_beg:y_end, ii], dsize=(width, height))

            if normalize:
                if normalize == "local_max":
                    images[slice_counter, :, :, 0] = img_patch / np.max(img_patch)
                elif normalize == "mean_std":
                    images[slice_counter, :, :, 0] = (img_patch-np.mean(img_patch))/np.std(img_patch)
                else:
                    raise ValueError('Please select a valid normalization')
            else:
                images[slice_counter, :, :, 0] = img_patch

            slice_counter += 1

        g = tf.Graph()
        with g.as_default():

            with tf.name_scope('inputs'):
                x = tf.placeholder(tf.float32, [None, width, height, n_channels])

            conv1 = conv_2d(x, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv1 = conv_2d(conv1, 32, 3, activation='relu', padding='same', regularizer="L2")
            pool1 = max_pool_2d(conv1, 2)

            conv2 = conv_2d(pool1, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv2 = conv_2d(conv2, 64, 3, activation='relu', padding='same', regularizer="L2")
            pool2 = max_pool_2d(conv2, 2)

            conv3 = conv_2d(pool2, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv3 = conv_2d(conv3, 128, 3, activation='relu', padding='same', regularizer="L2")
            pool3 = max_pool_2d(conv3, 2)

            conv4 = conv_2d(pool3, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv4 = conv_2d(conv4, 256, 3, activation='relu', padding='same', regularizer="L2")
            pool4 = max_pool_2d(conv4, 2)

            conv5 = conv_2d(pool4, 512, 3, activation='relu', padding='same', regularizer="L2")
            conv5 = conv_2d(conv5, 512, 3, activation='relu', padding='same', regularizer="L2")

            up6 = upsample_2d(conv5, 2)
            up6 = tflearn.layers.merge_ops.merge([up6, conv4], 'concat',axis=3)
            conv6 = conv_2d(up6, 256, 3, activation='relu', padding='same', regularizer="L2")
            conv6 = conv_2d(conv6, 256, 3, activation='relu', padding='same', regularizer="L2")

            up7 = upsample_2d(conv6, 2)
            up7 = tflearn.layers.merge_ops.merge([up7, conv3],'concat', axis=3)
            conv7 = conv_2d(up7, 128, 3, activation='relu', padding='same', regularizer="L2")
            conv7 = conv_2d(conv7, 128, 3, activation='relu', padding='same', regularizer="L2")

            up8 = upsample_2d(conv7, 2)
            up8 = tflearn.layers.merge_ops.merge([up8, conv2],'concat', axis=3)
            conv8 = conv_2d(up8, 64, 3, activation='relu', padding='same', regularizer="L2")
            conv8 = conv_2d(conv8, 64, 3, activation='relu', padding='same', regularizer="L2")

            up9 = upsample_2d(conv8, 2)
            up9 = tflearn.layers.merge_ops.merge([up9, conv1],'concat', axis=3)
            conv9 = conv_2d(up9, 32, 3, activation='relu', padding='same', regularizer="L2")
            conv9 = conv_2d(conv9, 32, 3, activation='relu', padding='same', regularizer="L2")

            pred = conv_2d(conv9, 2, 1,  activation='linear', padding='valid')

        with tf.Session(graph=g) as sess_test_seg:
            # Restore the model
            tf_saver = tf.train.Saver()
            tf_saver.restore(sess_test_seg, modelCkptSeg)

            for idx in range(images.shape[0]):
                im = np.reshape(images[idx, :, :], [1, width, height, n_channels])
                feed_dict = {x: im}
                pred_ = sess_test_seg.run(pred, feed_dict=feed_dict)
                percentileSeg = thresholdSeg * 100
                theta = np.percentile(pred_, percentileSeg)
                pred_bin = np.where(pred_ > theta, 1, 0)
                # Map predictions to original indices and size
                pred_bin = cv2.resize(
                    pred_bin[0, :, :, 0],
                    dsize=(y_end-y_beg, x_end-x_beg),
                    interpolation=cv2.INTER_NEAREST)
                pred3dFinal[idx, x_beg:x_end, y_beg:y_end,0] = pred_bin.astype('float64')

            pppp = True
            if pppp:
                pred3dFinal = self._post_processing(np.asarray(pred3dFinal))
            pred3d = [
                cv2.resize(
                    elem,
                    dsize=(image_data.shape[1], image_data.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ) for elem in pred3dFinal
            ]
            pred3d = np.asarray(pred3d)
            upsampled = np.swapaxes(np.swapaxes(pred3d, 1, 2), 0, 2)  # if Orient module applied, no need for this line(?)
            up_mask = nibabel.Nifti1Image(upsampled, img_nib.affine)

            # Save output mask
            save_file = self._gen_filename('out_file')
            nibabel.save(up_mask, save_file)

    def _extractLargestCC(self, image):
        """Function returning largest connected component of an object."""

        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(image, connectivity=4)
        sizes = stats[:, -1]
        max_label = 1
        # in case no segmentation
        if len(sizes) < 2:
            return image
        max_size = sizes[1]
        for i in range(2, nb_components):
            if sizes[i] > max_size:
                max_label = i
                max_size = sizes[i]
        largest_cc = np.zeros(output.shape)
        largest_cc[output == max_label] = 255
        return largest_cc.astype('uint8')

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = self._gen_filename('out_file')
        return outputs


class MultipleBrainExtractionInputSpec(BaseInterfaceInputSpec):
    """Class used to represent outputs of the MultipleBrainExtraction interface."""

    bids_dir = Directory(desc='Root directory', mandatory=True, exists=True)
    input_images = InputMultiPath(File(mandatory=True), desc='MRI Images')
    in_ckpt_loc = File(desc='Network_checkpoint for localization', mandatory=True)
    threshold_loc = traits.Float(0.49, desc='Threshold determining cutoff probability (0.49 by default)')
    in_ckpt_seg = File(desc='Network_checkpoint for segmentation', mandatory=True)
    threshold_seg = traits.Float(0.5, desc='Threshold determining cutoff probability (0.5 by default)')
    out_postfix = traits.Str("_brainMask", desc='Suffix of the automatically generated mask', usedefault=True)


class MultipleBrainExtractionOutputSpec(TraitedSpec):
    """Class used to represent outputs of the MultipleBrainExtraction interface."""

    masks = OutputMultiPath(File(), desc='Output masks')


class MultipleBrainExtraction(BaseInterface):
    """Runs on multiple images the automatic brain extraction module.

    It calls on a list of images the :class:`pymialsrtk.interfaces.preprocess.BrainExtraction.BrainExtraction` module
    that implements a brain extraction algorithm based on a 2D U-Net (Ronneberger et al. [1]_) using
    the pre-trained weights from Salehi et al. [2]_.

    References
    ------------
    .. [1] Ronneberger et al.; Medical Image Computing and Computer Assisted Interventions, 2015. `(link to paper) <https://arxiv.org/abs/1505.04597>`_
    .. [2] Salehi et al.; arXiv, 2017. `(link to paper) <https://arxiv.org/abs/1710.09338>`_

    See also
    ------------
    pymialsrtk.interfaces.preprocess.BrainExtraction

    """

    input_spec = MultipleBrainExtractionInputSpec
    output_spec = MultipleBrainExtractionOutputSpec

    def _run_interface(self, runtime):
        if len(self.inputs.input_images) > 0:
            for input_image in self.inputs.input_images:
                ax = BrainExtraction(bids_dir=self.inputs.bids_dir,
                                     in_file=input_image,
                                     in_ckpt_loc=self.inputs.in_ckpt_loc,
                                     threshold_loc=self.inputs.threshold_loc,
                                     in_ckpt_seg=self.inputs.in_ckpt_seg,
                                     threshold_seg=self.inputs.threshold_seg,
                                     out_postfix=self.inputs.out_postfix)
                ax.run()
        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['masks'] = glob(os.path.abspath("*.nii.gz"))
        return outputs

