import functions as fun
from SimpleITK import StatisticsImageFilter, ReadImage, GetArrayFromImage, GetImageFromArray
import numpy as np
import matplotlib.pyplot as plt

# # Main # #

if __name__ == '__main__':
    # # PRE-PROCESSING # #
    # --------------------------- #
    # Reading and processing various images,
    # filters and arrays
    # --------------------------- #

    # Reading reference images
    common_img_40 = ReadImage('./Common images & masks/common_40_image.nii.gz')
    common_img_41 = ReadImage('./Common images & masks/common_41_image.nii.gz')
    common_img_42 = ReadImage('./Common images & masks/common_42_image.nii.gz')

    # Reading moving images
    group_img_59 = ReadImage('./Group images/g3_59_image.nii.gz')
    group_img_60 = ReadImage('./Group images/g3_60_image.nii.gz')
    group_img_61 = ReadImage('./Group images/g3_61_image.nii.gz')

    # Creating necessary list
    ct_list = [group_img_59, group_img_60, group_img_61]

    # Reading hip bone masks
    hip_mask_img_59 = ReadImage('./Group images/g3_59_image_hipbone.nii.gz')
    hip_mask_img_60 = ReadImage('./Group images/g3_60_image_hipbone.nii.gz')
    hip_mask_img_61 = ReadImage('./Group images/g3_61_image_hipbone.nii.gz')

    # Reading femur bone masks
    femur_mask_img_59 = ReadImage('./Group images/g3_59_image_femur.nii.gz')
    femur_mask_img_60 = ReadImage('./Group images/g3_60_image_femur.nii.gz')
    femur_mask_img_61 = ReadImage('./Group images/g3_61_image_femur.nii.gz')

    # Reading common image masks
    common_mask_img_40 = ReadImage('./Common images & masks/common_40_mask.nii.gz')
    common_mask_img_41 = ReadImage('./Common images & masks/common_41_mask.nii.gz')
    common_mask_img_42 = ReadImage('./Common images & masks/common_42_mask.nii.gz')

    # Fetching common image arrays
    common_array_40 = GetArrayFromImage(common_img_40)
    common_array_41 = GetArrayFromImage(common_img_41)
    common_array_42 = GetArrayFromImage(common_img_42)

    # Fetching group image arrays
    group_array_59 = GetArrayFromImage(group_img_59)
    group_array_60 = GetArrayFromImage(group_img_60)
    group_array_61 = GetArrayFromImage(group_img_61)

    # ----------------- #
    # Group image 59 #
    # ----------------- #

    # Creating filters
    hip_filter_59 = StatisticsImageFilter()
    femur_filter_59 = StatisticsImageFilter()

    # Execute the filters
    hip_filter_59.Execute(hip_mask_img_59)
    femur_filter_59.Execute(femur_mask_img_59)

    # Fetching the labels from filters
    hip_label_59 = hip_filter_59.GetMaximum()
    femur_label_59 = femur_filter_59.GetMaximum()

    # Creating the moving masks with the labels
    mov_hip_mask_59 = (hip_mask_img_59 == hip_label_59)
    mov_femur_mask_59 = (femur_mask_img_59 == hip_label_59)

    # Combining moving masks to create one mask
    mov_mask_59 = mov_hip_mask_59 + mov_femur_mask_59

    # Fetching data from various masks
    hip_mask_array_59 = GetArrayFromImage(mov_hip_mask_59)
    femur_mask_array_59 = GetArrayFromImage(mov_femur_mask_59)

    # Combining data to create one data mask
    mov_mask_array_59 = hip_mask_array_59 + femur_mask_array_59

    # ----------------- #
    # Group image 60 #
    # ----------------- #

    # Creating filters
    hip_filter_60 = StatisticsImageFilter()
    femur_filter_60 = StatisticsImageFilter()

    # Execute the filters
    hip_filter_60.Execute(hip_mask_img_60)
    femur_filter_60.Execute(femur_mask_img_60)

    # Fetching the labels from filters
    hip_label_60 = hip_filter_60.GetMaximum()
    femur_label_60 = femur_filter_60.GetMaximum()

    # Creating the moving masks with the labels
    mov_hip_mask_60 = (hip_mask_img_60 == hip_label_60)
    mov_femur_mask_60 = (femur_mask_img_60 == hip_label_60)

    # Combining moving masks to create one mask
    mov_mask_60 = mov_hip_mask_60 + mov_femur_mask_60

    # Fetching arrays from various masks & images
    hip_mask_array_60 = GetArrayFromImage(mov_hip_mask_60)
    femur_mask_array_60 = GetArrayFromImage(mov_femur_mask_60)

    # Combining arrays to create one mask
    mov_mask_array_60 = hip_mask_array_60 + femur_mask_array_60

    # ----------------- #
    # Group image 61 #
    # ----------------- #

    # Creating filters
    hip_filter_61 = StatisticsImageFilter()
    femur_filter_61 = StatisticsImageFilter()

    # Execute the filters
    hip_filter_61.Execute(hip_mask_img_61)
    femur_filter_61.Execute(femur_mask_img_61)

    # Fetching the labels from filters
    hip_label_61 = hip_filter_61.GetMaximum()
    femur_label_61 = femur_filter_61.GetMaximum()

    # Creating the moving masks with the labels
    mov_hip_mask_61 = (hip_mask_img_61 == hip_label_61)
    mov_femur_mask_61 = (femur_mask_img_61 == hip_label_61)

    # Combining moving masks to create one mask
    mov_mask_61 = mov_hip_mask_61 + mov_femur_mask_61

    # Fetching data from various masks
    hip_mask_array_61 = GetArrayFromImage(mov_hip_mask_61)
    femur_mask_array_61 = GetArrayFromImage(mov_femur_mask_61)

    # Combining data to create one data mask
    mov_mask_array_61 = hip_mask_array_61 + femur_mask_array_61

    # Creating necessary list
    mask_list = [mov_mask_59, mov_mask_60, mov_mask_61]

    # ----------------- #
    # Labels #
    # ----------------- #

    # Fetching labels
    femur_label = 1
    hipbone_label = 3

    # Pre-processing masks (like above) #
    # ----------------- #
    # Common mask 40 #
    # ----------------- #

    mask_40_filter = StatisticsImageFilter()
    mask_40_filter.Execute(common_mask_img_40)
    hipbone_mask_40 = (common_mask_img_40 == hipbone_label)
    femur_mask_40 = (common_mask_img_40 == femur_label)
    mask_img_40 = hipbone_mask_40 + femur_mask_40
    new_mask_array_40 = GetArrayFromImage(mask_img_40)

    # ----------------- #
    # Common mask 41 #
    # ----------------- #

    mask_41_filter = StatisticsImageFilter()
    mask_41_filter.Execute(common_mask_img_41)
    hipbone_mask_41 = (common_mask_img_41 == hipbone_label)
    femur_mask_41 = (common_mask_img_41 == femur_label)
    mask_img_41 = hipbone_mask_41 + femur_mask_41
    new_mask_array_41 = GetArrayFromImage(mask_img_41)

    # ----------------- #
    # Common mask 42 #
    # ----------------- #

    mask_42_filter = StatisticsImageFilter()
    mask_42_filter.Execute(common_mask_img_42)
    hipbone_mask_42 = (common_mask_img_42 == hipbone_label)
    femur_mask_42 = (common_mask_img_42 == femur_label)
    mask_img_42 = hipbone_mask_42 + femur_mask_42
    new_mask_array_42 = GetArrayFromImage(mask_img_42)

    #  # Done pre-processing #  #
    # ------------------------------------------------------------ #

    # ----------------- #
    # SEGMENTATION #
    # ----------------- #
    seg_array_1 = fun.seg_atlas(common_img_40, ct_list, mask_list)
    seg_array_2 = fun.seg_atlas(common_img_41, ct_list, mask_list)
    seg_array_3 = fun.seg_atlas(common_img_42, ct_list, mask_list)

    # Get images from segmentation arrays
    seg_img_40 = GetImageFromArray(seg_array_1)
    seg_img_41 = GetImageFromArray(seg_array_2)
    seg_img_42 = GetImageFromArray(seg_array_3)

    # Get images from mask arrays
    mask_img_40 = GetImageFromArray(new_mask_array_40)
    mask_img_41 = GetImageFromArray(new_mask_array_41)
    mask_img_42 = GetImageFromArray(new_mask_array_42)

    # ----------------- #
    # EVALUATION #
    # ----------------- #

    # Execute similarity functions
    fun.distances(mask_img_40, seg_img_40)
    fun.distances(mask_img_41, seg_img_41)
    fun.distances(mask_img_42, seg_img_42)

    # ----------------- #
    # PLOTTING #
    # ----------------- #

    # Common image 40
    plt.figure(figsize=(20, 50))
    plt.subplot(131)
    plt.imshow(common_array_40[75, :, :], cmap='gray')  # fixed image
    plt.title('Reference image')
    plt.subplot(132)
    plt.imshow(seg_array_1[75, :, :], cmap='gray')  # fixed image
    plt.title('Segmentation')
    plt.subplot(133)
    plt.imshow(common_array_40[75, :, :], cmap='gray')  # fixed image
    plt.imshow(seg_array_1[75, :, :], cmap='Blues', alpha=0.5)  # moving image
    plt.title('Reference image + Segmentation')
    plt.show()

    # Common image 41
    plt.figure(figsize=(20, 50))
    plt.subplot(131)
    plt.imshow(common_array_41[75, :, :], cmap='gray')  # fixed image
    plt.title('Reference image')
    plt.subplot(132)
    plt.imshow(seg_array_2[75, :, :], cmap='gray')  # segmentation image
    plt.title('Segmentation')
    plt.subplot(133)
    plt.imshow(common_array_41[75, :, :], cmap='gray')  # fixed image
    plt.imshow(seg_array_2[75, :, :], cmap='Blues', alpha=0.5)  # moving image
    plt.title('Reference image + Segmentation')
    plt.show()

    # Common image 42
    plt.figure(figsize=(20, 50))
    plt.subplot(131)
    plt.imshow(common_array_42[75, :, :], cmap='gray')  # fixed image
    plt.title('Reference image')
    plt.subplot(132)
    plt.imshow(seg_array_3[75, :, :], cmap='gray')  # fixed image
    plt.title('Segmentation')
    plt.subplot(133)
    plt.imshow(common_array_42[75, :, :], cmap='gray')  # fixed image
    plt.imshow(seg_array_3[75, :, :], cmap='Blues', alpha=0.5)  # moving image
    plt.title('Reference image + Segmentation')
    plt.show()

    # ----------------- #
    # CLASSIFICATION #
    # ----------------- #

    # Creating the vectors needed for labels
    # The selected intervall of slices are hand picked
    vector_59 = np.zeros(512, dtype=np.uint8)
    vector_59[247:253] = 1
    vector_60 = np.zeros(512, dtype=np.uint8)
    vector_60[250:255] = 1
    vector_61 = np.zeros(512, dtype=np.uint8)
    vector_61[242:247] = 1

    # Creating necessary list
    vector_list = [vector_59, vector_60, vector_61]

    # Training the random forest and finding the slice with highest
    # probability of containing the pubic symphysis
    trained_tree = fun.train_classifier(ct_list, vector_list)
    fun.slice_probability(common_img_40, trained_tree)
