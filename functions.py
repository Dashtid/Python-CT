import SimpleITK as sitk
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# # Estimation function # #
# --------------------------- #
# Linear registration function
# --------------------------- #

# --- Input --- #
# im_ref : The common image [numpy.ndarray]
# im_mov : The group image  [numpy.ndarray]
# mov_mask : List of GROUP masks [list]
# show_parameters : If you want to see the parameters, false by default [boolean]


# --- Output --- #
# lin_xfm : Estimated transformation parameters [itk.simple.Transform]

def est_lin_transf(im_ref, im_mov, mov_mask=None, show_parameters=False):
    initial_transform = sitk.CenteredTransformInitializer(im_ref, im_mov, sitk.ScaleSkewVersor3DTransform(),
                                                          sitk.CenteredTransformInitializerFilter.MOMENTS)

    # Initialize registration
    lin_transformation = sitk.ImageRegistrationMethod()

    # Set metrics
    lin_transformation.SetMetricAsMeanSquares()
    lin_transformation.SetMetricSamplingStrategy(lin_transformation.RANDOM)
    lin_transformation.SetMetricSamplingPercentage(0.01)

    # Set mask
    if mov_mask:
        lin_transformation.SetMetricMovingMask(mov_mask)

    # Gradient Descent optimizer
    lin_transformation.SetOptimizerAsGradientDescent(learningRate=1, numberOfIterations=400,
                                                     convergenceMinimumValue=1e-6, convergenceWindowSize=10)
    lin_transformation.SetOptimizerScalesFromPhysicalShift()

    # Set the initial transformation
    lin_transformation.SetInitialTransform(initial_transform)

    # Switching to preferred variable
    lin_xfm = lin_transformation

    if show_parameters:
        print(lin_xfm)

    return lin_xfm


# # Estimation function # #
# --------------------------- #
# Non-linear 'Demons' registration function
# --------------------------- #

# --- Input --- #
# im_ref : The common image [numpy.ndarray]
# fixed_mask : The mask of common image, default is None [numpy.ndarray]
# show_parameters : If you want to see the parameters, false by default [boolean]


# --- Output --- #
# nl_xfm : Estimated transformation parameters [itk.simple.Transform]

def est_nl_transf(im_ref, fixed_mask=None, show_parameters=False):
    # Initialize the registration
    reg_method = sitk.ImageRegistrationMethod()

    # Create initial identity transformation.
    transform_to_displacement_field_filter = sitk.TransformToDisplacementFieldFilter()
    transform_to_displacement_field_filter.SetReferenceImage(im_ref)
    initial_transform = sitk.DisplacementFieldTransform(
        transform_to_displacement_field_filter.Execute(sitk.Transform()))

    # Regularization. The update field refers to fluid regularization; the total field to elastic regularization.
    initial_transform.SetSmoothingGaussianOnUpdate(varianceForUpdateField=0, varianceForTotalField=1.5)

    # Set the initial transformation
    reg_method.SetInitialTransform(initial_transform)

    # Set Demons registration
    reg_method.SetMetricAsDemons(intensityDifferenceThreshold=0.001)

    # Evaluate the metrics only in the mask
    if fixed_mask is not None:
        reg_method.SetMetricFixedMask(fixed_mask)

    # Set a linear interpolator
    reg_method.SetInterpolator(sitk.sitkLinear)

    # Set a gradient descent optimizer
    reg_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=10, convergenceMinimumValue=1e-6,
                                             convergenceWindowSize=10)
    reg_method.SetOptimizerScalesFromPhysicalShift()

    # Switching to the preferred variable
    nl_xfm = reg_method

    if show_parameters:
        print(nl_xfm)

    return nl_xfm

# # Application function # #
# --------------------------- #
# Executes either the linear or the non-linear function
# --------------------------- #

# --- Input --- #
# im_ref : The common image [numpy.ndarray]
# im_mov : The group image  [numpy.ndarray]
# trafo : The chosen transformation [numpy.ndarray]
# show_parameters : If you want to see the parameters, false by default [boolean]


# --- Output --- #
# final_image : Returns the registered image [numpy.ndarray]

def apply_transf(im_ref, im_mov, trafo, show_parameters=False):
    # Perform registration (Executes it)
    transf = trafo.Execute(sitk.Cast(im_ref, sitk.sitkFloat32), sitk.Cast(im_mov, sitk.sitkFloat32))

    if show_parameters:
        print(transf)
        print("--------")
        print("Optimizer stop condition: {0}".format(trafo.GetOptimizerStopConditionDescription()))
        print("Number of iterations: {0}".format(trafo.GetOptimizerIteration()))
        print("--------")

    return transf


# # Atlas segmentation function # #
# --------------------------- #
# Atlas-based segmentation using the CT images in 'ct_list'
# and corresponding segmentation masks from 'seg_list'.
# After that, majority voting to return a segmentation mask.
# --------------------------- #

# --- Input --- #
# common_img : The chosen COMMON image [sitk-image]
# ct_list : List of GROUP images [list]
# seg_list : List of GROUP masks [list]

# --- Output --- #
# segmented_array : The segmentation as an array [numpy.ndarray]

def seg_atlas(common_img, ct_list, seg_list):
    """
    Perform atlas-based segmentation on the reference image using provided CT and mask lists.

    Args:
        reference_img: SimpleITK.Image, the reference image to segment.
        ct_list: list of SimpleITK.Image, moving images.
        mask_list: list of SimpleITK.Image, corresponding masks for moving images.

    Returns:
        np.ndarray: Segmentation result as a numpy array.
    """
    # Creating the necessary lists
    seg = []
    image_list = []

    # # REGISTRATION # #
    for i in range(len(ct_list)):
        # Adjusting the settings and applying
        trafo_settings = est_lin_transf(common_img, ct_list[i], mov_mask=seg_list[i], show_parameters=False)
        final_trafo = apply_transf(common_img, ct_list[i], trafo_settings)

        # Perform registration on mask image
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(common_img)
        resampler.SetInterpolator(sitk.sitkLinear)

        resampler.SetTransform(final_trafo)
        resampled_mask = resampler.Execute(seg_list[i])

        resampled_mask_data = sitk.GetArrayFromImage(resampled_mask)
        seg.append(resampled_mask_data)

    # # MAJORITY VOTING # #
    for i in range(len(seg)):
        for j in range(i + 1, len(seg)):
            arr1 = np.transpose(np.nonzero(seg[i]))
            arr2 = np.transpose(np.nonzero(seg[j]))

            # Filling two lists
            arr1list = [tuple(e) for e in arr1.tolist()]
            arr2list = [tuple(e) for e in arr2.tolist()]

            # Sorting both lists
            arr1list.sort()
            arr2list.sort()

            # Creating necessary list & sorting
            intersections = list(set(arr1list).intersection(arr2list))
            intersections.sort()

            image_list.append(intersections)
    # Creating a list which contains the indexes of intersecting voxels
    intersection_list = list(set(image_list[0]) | set(image_list[1]) | set(image_list[2]))

    # Sorting the list
    intersection_list.sort()

    # Fetches array from image
    image_array = sitk.GetArrayFromImage(common_img)

    # Creates an array for the points and fills it using indexes
    segmented_array = np.zeros(shape=image_array.shape, dtype=np.uint8)
    for x, y, z in intersection_list:
        segmented_array[x, y, z] = 1

    return segmented_array


# # Similarity function # #
# --------------------------- #
# Calculates the following distances between images:
# 1. Jaccard coef.
# 2. Dice coef.
# 3. Hausdorff distance
# --------------------------- #

# --- Input --- #
# mask_img : The mask image [sikt-image]
# seg_img: The segmented image [sikt-image]

# --- Output --- #
# None

def distances(mask_img, seg_img):
    """
    Compute similarity metrics between mask and segmentation.

    Args:
        mask_img: SimpleITK.Image, ground truth mask.
        seg_img: SimpleITK.Image, predicted segmentation.

    Returns:
        dict: Dictionary of similarity metrics.
    """
    # Creating the necessary filters
    hausdorff = sitk.HausdorffDistanceImageFilter()
    overlap = sitk.LabelOverlapMeasuresImageFilter()

    # Execute filters
    hausdorff.Execute(mask_img, seg_img)
    overlap.Execute(mask_img, seg_img)

    # Fetching the distances and appending to distance list
    # Jaccard coef.
    jaccard = overlap.GetJaccardCoefficient()

    # Dice coef.
    dice = overlap.GetDiceCoefficient()

    # Hausdorff distance
    hausdorff_distance = hausdorff.GetHausdorffDistance()

    # Printing out the distances for user
    print('The Hausdorff distance: {}'.format(
        hausdorff_distance))
    print('The Dice coefficient: {}'.format(dice))
    print('The Jaccard coefficient: {}'.format(jaccard))

    return None


# # Classifier Function # #
# --------------------------- #
# Trains a random forest classifier by reading 2d images and comparing
# them to a vector which has labels that correspond to if it contains
# the pubic symphysis. The labels are binary.
# --------------------------- #

# --- Input --- #
# slice_list : List of 2D slice images [list]
# vector_list : List of vectors with binary labels [list]

# --- Output --- #
# trained_forest : Trained random forest classifier [sklearn.ensemble.forest.RandomForestClassifier]

def train_classifier(slice_list, vector_list):
    """
    Train a random forest classifier to predict slice labels.

    Args:
        ct_list: list of SimpleITK.Image, CT images.
        vector_list: list of np.ndarray, label vectors for each CT.

    Returns:
        trained_model: Trained classifier.
    """
    # Creating necessary list
    x_train_list = []

    # Reading in input data
    for image in slice_list:

        # Fetching arrays
        image_array = sitk.GetArrayFromImage(image)

        # Resizing
        image_array.resize((512, 512, 512))

        for z in range(image_array.shape[2]):
            x_train_list.append(image_array[:, :, z].flatten())
    x_train = np.asarray(x_train_list, dtype=np.uint8)

    # Reading in training labels
    y_train = None
    for i in range(0, len(vector_list)):
        if i == 0:
            y_train = vector_list[i]
        else:
            y_train = np.concatenate([y_train, vector_list[i]])

    # Train classifier
    trained_forest = RandomForestClassifier(n_estimators=150)
    trained_forest.fit(x_train, y_train)

    return trained_forest


# # Classifier Function # #
# --------------------------- #
# Utilizes a trained random forest classifier by reading CT image and prints
# which slice has the highest probability of containing the pubic symphysis.
# --------------------------- #

# --- Input --- #
# ct_image : List of 2D axial slice images [list]
# classifier : Trained random forest classifier [sklearn.ensemble.forest.RandomForestClassifier]

# --- Output --- #
# None

def slice_probability(ct_image, classifier):
    """
    Predict the probability of each slice containing the pubic symphysis.

    Args:
        reference_img: SimpleITK.Image, the reference image.
        trained_tree: trained classifier.

    Returns:
        np.ndarray: Probability scores for each slice.
    """
    # Creating necessary lists
    test_list = []
    max_list = []

    # Convert image to numpy array & resize
    im_array = sitk.GetArrayFromImage(ct_image)
    im_array.resize((512, 512, 512))

    for z in range(im_array.shape[2]):
        test_list.append(im_array[:, :, z].flatten())
    test_array = np.asarray(test_list, dtype=np.uint8)

    # Predict probabilities for each slice
    probabilities = classifier.predict_proba(test_array)

    # Fetching array with maximum probabilities
    max = np.amax(probabilities, axis=0)[1]

    for i, prob in enumerate(probabilities):
        if prob[1] == max:
            max_list.append(i)

    # Print result to user
    if len(max_list) == 1:
        print("Slice {} has highest probability which is: {}".format(max_list[0], max))
    else:
        print("Slices {} have the highest probability which is: {}".format(max_list, max))

    return None
