import functions as fun
from SimpleITK import StatisticsImageFilter, ReadImage, GetArrayFromImage, GetImageFromArray
import numpy as np
import matplotlib.pyplot as plt

# Main script for CT image processing, segmentation, evaluation, and classification

if __name__ == '__main__':
    # --------------------------- #
    # PRE-PROCESSING
    # --------------------------- #

    # Read reference images
    common_img_40 = ReadImage('./Common images & masks/common_40_image.nii.gz')
    common_img_41 = ReadImage('./Common images & masks/common_41_image.nii.gz')
    common_img_42 = ReadImage('./Common images & masks/common_42_image.nii.gz')

    # Read moving images
    group_img_59 = ReadImage('./Group images/g3_59_image.nii.gz')
    group_img_60 = ReadImage('./Group images/g3_60_image.nii.gz')
    group_img_61 = ReadImage('./Group images/g3_61_image.nii.gz')
    ct_list = [group_img_59, group_img_60, group_img_61]

    # Read hip and femur bone masks for each group image
    hip_mask_imgs = [
        ReadImage(f'./Group images/g3_{i}_image_hipbone.nii.gz') for i in [59, 60, 61]
    ]
    femur_mask_imgs = [
        ReadImage(f'./Group images/g3_{i}_image_femur.nii.gz') for i in [59, 60, 61]
    ]

    # Read common image masks
    common_mask_imgs = [
        ReadImage(f'./Common images & masks/common_{i}_mask.nii.gz') for i in [40, 41, 42]
    ]

    # Convert images to arrays for plotting
    common_arrays = [GetArrayFromImage(img) for img in [common_img_40, common_img_41, common_img_42]]
    group_arrays = [GetArrayFromImage(img) for img in [group_img_59, group_img_60, group_img_61]]

    # --------------------------- #
    # MASK PROCESSING
    # --------------------------- #

    # Process moving masks for each group image
    mask_list = []
    for hip_mask_img, femur_mask_img in zip(hip_mask_imgs, femur_mask_imgs):
        hip_filter = StatisticsImageFilter()
        femur_filter = StatisticsImageFilter()
        hip_filter.Execute(hip_mask_img)
        femur_filter.Execute(femur_mask_img)
        hip_label = hip_filter.GetMaximum()
        # Use hip_label for both hip and femur masks as in original code
        mov_hip_mask = (hip_mask_img == hip_label)
        mov_femur_mask = (femur_mask_img == hip_label)
        mov_mask = mov_hip_mask + mov_femur_mask
        mask_list.append(mov_mask)

    # --------------------------- #
    # COMMON MASKS PROCESSING
    # --------------------------- #

    femur_label = 1
    hipbone_label = 3
    new_mask_arrays = []
    for common_mask_img in common_mask_imgs:
        mask_filter = StatisticsImageFilter()
        mask_filter.Execute(common_mask_img)
        hipbone_mask = (common_mask_img == hipbone_label)
        femur_mask = (common_mask_img == femur_label)
        mask_img = hipbone_mask + femur_mask
        new_mask_arrays.append(GetArrayFromImage(mask_img))

    # --------------------------- #
    # SEGMENTATION
    # --------------------------- #

    seg_arrays = [
        fun.seg_atlas(common_img, ct_list, mask_list)
        for common_img in [common_img_40, common_img_41, common_img_42]
    ]
    seg_imgs = [GetImageFromArray(seg_array) for seg_array in seg_arrays]
    mask_imgs = [GetImageFromArray(mask_array) for mask_array in new_mask_arrays]

    # --------------------------- #
    # EVALUATION
    # --------------------------- #

    for mask_img, seg_img in zip(mask_imgs, seg_imgs):
        fun.distances(mask_img, seg_img)

    # --------------------------- #
    # PLOTTING
    # --------------------------- #

    for i, (common_array, seg_array) in enumerate(zip(common_arrays, seg_arrays)):
        plt.figure(figsize=(20, 50))
        plt.subplot(131)
        plt.imshow(common_array[75, :, :], cmap='gray')
        plt.title('Reference image')
        plt.subplot(132)
        plt.imshow(seg_array[75, :, :], cmap='gray')
        plt.title('Segmentation')
        plt.subplot(133)
        plt.imshow(common_array[75, :, :], cmap='gray')
        plt.imshow(seg_array[75, :, :], cmap='Blues', alpha=0.5)
        plt.title('Reference image + Segmentation')
        plt.show()

    # --------------------------- #
    # CLASSIFICATION
    # --------------------------- #

    # Create label vectors for slices (intervals are hand-picked)
    vector_list = [
        np.pad(np.ones(6, dtype=np.uint8), (247, 512-253), 'constant'),
        np.pad(np.ones(5, dtype=np.uint8), (250, 512-255), 'constant'),
        np.pad(np.ones(5, dtype=np.uint8), (242, 512-247), 'constant')
    ]

    # Train random forest and find slice with highest probability of pubic symphysis
    trained_tree = fun.train_classifier(ct_list, vector_list)
    fun.slice_probability(common_img_40, trained_tree)
