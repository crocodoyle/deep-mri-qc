import numpy as np
import matplotlib.pyplot as plt


data_dir = '/data1/users/adoyle/'

def normalize(image, mask, target_image, target_mask):

    img_shape = image.shape

    n_landmarks = 10

    valid_orig_image = image[~mask]
    valid_taget_image = target_image[~target_mask]

    valid_orig_flat = []
    for x in range(img_shape[0]):
        for y in range(img_shape[1]):
            for z in range(img_shape[2]):
                if mask[x, y, z] == 0:
                    valid_orig_flat.append(image[x, y, z])

    hist_original = np.histogram(valid_orig_flat, bins=256)
    fig = plt.figure()

    plt.bar(hist_original[0], hist_original[1][:-1])
    plt.title('original histogram')
    plt.savefig(data_dir + 'orig_hist.png')

    p2_orig, p98_orig = np.percentile(valid_orig_image, (2, 98))
    orig_landmarks = np.arange(p2_orig, p98_orig, (p98_orig - p2_orig)/n_landmarks)

    print('landmarks:', orig_landmarks)

    p2_target, p98_target = np.percentile(valid_taget_image, (2, 98))
    target_landmarks = np.arange(p2_target, p98_target, (p98_target - p2_target)/n_landmarks)

    hist_target = np.histogram(target_image)

    fig = plt.figure()

    plt.bar(hist_target[0], hist_target[1][:-1])
    plt.title('target histogram')
    plt.savefig(data_dir + 'target_hist.png')

    rescaled_image = np.copy(image)

    for i in range(n_landmarks-1):
        low_orig, high_orig = orig_landmarks[i], orig_landmarks[i+1]
        low_target, high_target = target_landmarks[i], target_landmarks[i+1]

        b_orig = low_orig
        a_orig = high_orig - low_orig

        b_target = low_target
        a_target = high_target - low_target

        b_transform = b_target - b_orig
        a_transform = a_target - a_orig

        for x in range(img_shape[0]):
            for y in range(img_shape[1]):
                for z in range(img_shape[2]):
                    if mask[x, y, z] == 0:
                        pixel = image[x, y, z]

                        if pixel >= low_orig and pixel < high_orig:
                            rescaled_image[x, y, z] = b_transform + pixel*a_transform
    return rescaled_image


if __name__ == '__main__':

    import nibabel as nib


    target_size = (192, 256, 192)

    orig = nib.load(data_dir + '50002.mnc').get_data()
    target = nib.load(data_dir + 'ds030.nii.gz').get_data()

    mask = nib.load(data_dir + 'mni_icbm152_t1_tal_nlin_asym_09a_mask.mnc').get_data()

    from make_datasets import resize_image_with_crop_or_pad

    orig = resize_image_with_crop_or_pad(orig, target_size)
    target = resize_image_with_crop_or_pad(target, target_size)
    mask = resize_image_with_crop_or_pad(mask, target_size)
    target_mask = mask

    returned_image = normalize(orig, mask, target, target_mask)


    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.imshow(orig[96, :, :])
    ax2.imshow(target[96, : ,:])
    ax3.imshow(mask[96, :, :])

    plt.savefig(data_dir + 'nyul_results.png')