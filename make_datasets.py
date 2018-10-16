import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess, pickle, h5py

from sklearn.neighbors import KDTree
from skimage.exposure import adjust_sigmoid

import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.exposure import equalize_hist

from multiprocessing import Pool, Process

from nipype.interfaces.ants import Registration

data_dir = '/data1/users/adoyle/'
output_dir = '/data1/users/adoyle/deepqc/'

cores = 12

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

exemplar_file = data_dir + '/PING/p0086_20100316_193008_2_mri.mnc'

atlas = data_dir + '/mni_icbm152_t1_tal_nlin_asym_09a.mnc'
atlas_mask = data_dir + '/mni_icbm152_t1_tal_nlin_asym_09a_mask.mnc'

mri_sites = ['IBIS', 'PING', 'PITT', 'OLIN', 'OHSU', 'SDSU', 'TRINITY', 'UM', 'USM', 'YALE', 'CMU', 'LEUVEN', 'KKI',
             'NYU', 'STANFORD', 'UCLA', 'MAX_MUN', 'CALTECH', 'SBL', 'ds030']

target_size = (189, 233, 197)

# taken from DLTK
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) + 0.000001)
    return ret

# taken from DLTK
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


def count_ping(input_path, label_file):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        num_subjects = 0
        for line in list(qc_reader):
            try:
                label = line[1]
                if len(label) < 1:
                    raise Exception()
                t1_filename = line[0][:-4] + '.mnc'
                t1_data = nib.load(input_path + '/resampled/' + t1_filename)
                num_subjects += 1
            except:
                print('missing', t1_filename)

    return num_subjects

def count_ibis(input_path, label_file):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        num_subjects = 0
        for line in list(qc_reader):
            try:
                label = line[1]
                if len(label) < 1:
                    raise Exception()
                t1_filename = line[0][0:-4] + '.mnc'
                t1_data = nib.load(input_path + '/resampled/' + t1_filename)
                num_subjects += 1
            except:
                print('missing', t1_filename)

    return num_subjects

def count_abide(input_path, label_file):

    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        num_subjects = 0
        for line in list(qc_reader):
            try:
                label1 = line[2]
                label2 = line[3]

                if len(label1) + len(label2) < 1:
                    raise Exception()
                t1_filename = line[0] + '.mnc'
                t1_data = nib.load(input_path + '/resampled/' + t1_filename)
                num_subjects += 1
            except:
                print('missing', t1_filename)

    return num_subjects


def count_ds030(input_path, label_file):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        num_subjects = 0
        for line in list(qc_reader):
            try:
                label = line[8]
                if len(label) < 1:
                    raise Exception()
                t1_filename = line[0] + '.nii.gz'
                t1_data = nib.load(input_path + t1_filename)
                num_subjects += 1
            except:
                print('missing', t1_filename)

    return num_subjects


def make_ping(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        lines = list(qc_reader)
        input_paths = [input_path] * len(lines)

        mask = nib.load(atlas_mask).get_data()
        big_mask = resize_image_with_crop_or_pad(mask, target_size, mode='constant')

        index = subject_index
        for line, input_path in zip(lines, input_paths):
            returned_index = make_ping_subject(line, index, input_path, f, big_mask)
            if not returned_index == -1:
                index += 1

    return index


def make_ping_subject(line, subject_index, input_path, f, mask):
    try:
        t1_filename = line[0][:-4] + '.mnc'

        label = int(line[1])  # 0, 1, or 2
        comment = line[2]

        # register_MINC(input_path + t1_filename, atlas, input_path + '/resampled/' + t1_filename)

        one_hot = [0, 0]

        if label >= 1:
            one_hot = [0, 1]
        else:
            one_hot = [1, 0]

        f['qc_label'][subject_index, :] = one_hot
        # print(t1_filename, one_hot)
        t1_data = nib.load(input_path + '/resampled/' + t1_filename).get_data()

        if not t1_data.shape == target_size:
            # print('resizing from', t1_data.shape)
            t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')
        # f['comment'][subject_index] = comment

        t1_data = equalize_hist(t1_data, mask=mask)

        f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)
        f['dataset'][subject_index] = 'PING'
        f['filename'][subject_index] = t1_filename

        # plt.imshow(t1_data[96, ...])
        # plt.axis('off')
        # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

        return subject_index
    except Exception as e:
        print('Error:', e)
        return -1


def make_ibis(input_path, f, label_file, subject_index):

    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)

        lines = list(qc_reader)
        input_paths = [input_path] * len(lines)

        mask = nib.load(atlas_mask).get_data()
        big_mask = resize_image_with_crop_or_pad(mask, target_size, mode='constant')

        index = subject_index
        for line, input_path in zip(lines, input_paths):
            returned_index = make_ibis_subject(line, index, input_path, f, big_mask)
            if not returned_index == -1:
                index += 1

    return index


def make_ibis_subject(line, subject_index, input_path, f, mask):
    try:
        t1_filename = line[0][0:-4] + '.mnc'

        label = int(line[1])  # 0, 1, or 2

        # register_MINC(input_path + t1_filename, atlas, input_path + '/resampled/' + t1_filename)

        one_hot = [0, 0]

        if label >= 1:
            one_hot = [0, 1]
        else:
            one_hot = [1, 0]

        f['qc_label'][subject_index, :] = one_hot
        # print(t1_filename, one_hot)
        t1_data = nib.load(input_path + '/resampled/' + t1_filename).get_data()

        if not t1_data.shape == target_size:
            # print('resizing from', t1_data.shape)
            t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

        # t1_data = equalize_hist(t1_data, mask=mask)

        f['MRI'][subject_index, ...] = adjust_sigmoid(normalise_zero_one(t1_data))
        f['dataset'][subject_index] = 'IBIS'
        f['filename'][subject_index] = t1_filename

        # plt.imshow(t1_data[96, ...])
        # plt.axis('off')
        # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

        return subject_index
    except Exception as e:
        print('Error:', e)
        return -1


def make_abide(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        lines = list(qc_reader)
        input_paths = [input_path] * len(lines)

        mask = nib.load(atlas_mask).get_data()
        big_mask = resize_image_with_crop_or_pad(mask, target_size, mode='constant')

        index = subject_index
        for line, input_path in zip(lines, input_paths):
            returned_index = make_abide_subject(line, index, input_path, f, big_mask)
            if not returned_index == -1:
                index += 1

    return index

def make_abide_subject(line, subject_index, input_path, f, mask):
    try:
        t1_filename = line[0] + '.mnc'

        # register_MINC(input_path + t1_filename, atlas, input_path + '/resampled/' + t1_filename)

        one_hot = [0, 0, 0]

        total_labels = 0
        if len(line[2]) > 0:
            label1 = int(line[2]) + 1  # -1, 0, or 1
            one_hot[label1] += 1
            total_labels += 1
        if len(line[3]) > 0:
            label2 = int(line[3]) + 1
            one_hot[label2] += 1
            total_labels += 1
        # if len(line[4]) > 0:
        #     label3 = int(line[4]) + 1
        #     one_hot[label3] += 1
        #     total_labels += 1

        one_hot = [one_hot[0], one_hot[1] + one_hot[2]] # map doubtful and accept to one label

        one_hot = np.multiply(one_hot, 1 / total_labels)

        # for one in one_hot:
        #     if one > 0.1 and one < 0.9:
        #         print('Excluding ' + line[0] + ' because QC inconsistent')
        #         return -1

        f['qc_label'][subject_index, :] = np.argmax(one_hot)
        # print(t1_filename, one_hot)
        t1_data = nib.load(input_path + '/resampled/' + t1_filename).get_data()

        if not t1_data.shape == target_size:
            t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

        # t1_data = equalize_hist(t1_data, mask=mask)

        f['MRI'][subject_index, ...] = adjust_sigmoid(normalise_zero_one(t1_data))
        f['dataset'][subject_index] = line[1]
        f['filename'][subject_index] = t1_filename

        # plt.imshow(t1_data[96, ...])
        # plt.axis('off')
        # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

        return subject_index
    except Exception as e:
        print('Error:', e)
        return -1


def make_ds030(input_path, f, label_file, subject_index):
    with open(os.path.join(input_path, label_file), 'r') as label_file:
        qc_reader = csv.reader(label_file)

        lines = list(qc_reader)[1:]
        input_paths = [input_path] * len(lines)

        mask = nib.load(atlas_mask).get_data()
        big_mask = resize_image_with_crop_or_pad(mask, target_size, mode='constant')

        index = subject_index
        for line, input_path in zip(lines, input_paths):
            # print('starting subject:', line)
            returned_index = make_ds030_subject(line, index, input_path, f, big_mask)
            if not returned_index == -1:
                index += 1

    return index


def make_ds030_subject(line, subject_index, input_path, f, mask):
    try:
        t1_filename = line[0] + '.nii.gz'
        label = line[8]

        if len(label) > 0:
            t1 = nib.load(input_path + t1_filename)
            atlas_image = nib.load(atlas)

            t1_resampled = resample_from_to(t1, atlas_image)
            t1_data = t1_resampled.get_data()

            if not t1_data.shape == target_size:
                print('resizing from', t1_data.shape)
                t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

            # t1_data = equalize_hist(t1_data, mask=mask)

            f['MRI'][subject_index, ...] = adjust_sigmoid(normalise_zero_one(t1_data))

            one_hot = [0, 0]

            if 'ok' in label:
                one_hot = [0, 1]
            elif 'maybe' in label:
                one_hot = [0, 1]
            elif 'exclude' in label:
                one_hot = [1, 0]
            else:
                raise Exception

            f['qc_label'][subject_index, :] = np.argmax(one_hot)
            f['dataset'][subject_index] = 'ds030'
            f['filename'][subject_index] = t1_filename

            # plt.imshow(t1_data[96, ...])
            # plt.axis('off')
            # plt.savefig(output_dir + t1_filename[:-4] + '.png', bbox_inches='tight', cmap='gray')

    except Exception as e:
        print('Error:', e)
        return -1

    return subject_index


def make_abide_surfaces(path, label_file):
    patient_data = {}

    for index, filename in enumerate(os.listdir(path + '/T1_downsampled/')):
        if '.mnc' in filename:
            patient_id = filename[:-4]
            patient_data[patient_id] = {}

            print(filename, patient_id)
            patient_data[patient_id]['index'] = index
        else:
            index -= 1

    total_subjects = index + 1
    print('total of ', total_subjects, 'downsampled T1s')

    f = h5py.File(output_path + 'abide.hdf5', 'w')
    # f.create_dataset('images', (total_subjects, 181, 217, 181, 3), dtype='float32')  # t1, gradient magnitude, surface distance
    f.create_dataset('surfacepoints', (total_subjects, 40962*2, 3))
    # f.create_dataset('filenames', (total_subjects,), dtype=h5py.special_dtype(vlen=unicode))
    f.create_dataset('labels', (total_subjects,), dtype='bool')


    # load images and compute gradient
    # for filename in os.listdir(path + '/T1_downsampled/'):
    #     if '.mnc' in filename:
    #         patient_id = filename.split('.')[0]
    #
    #         i = patient_data[patient_id]['index']
    #         print(i, 'of', len(os.listdir(path + '/T1_downsampled/')))
    #
    #         img = nib.load(os.path.join(path + '/T1_downsampled/', filename)).get_data()  # load image data
    #
    #         f['images'][i, :, :, :, 0] = img
    #         f['images'][i, :, :, :, 1] = np.sum(np.gradient(img), axis=0)
    #     else:
    #         print(filename, 'should not be here')

    # extract surfaces from combined left/right surface objs
    for filename in os.listdir(path + '/surfaces/'):
        patient_id = filename.split('+')[1]

        i = patient_data[patient_id]['index']
        surface_obj = open(path + '/surfaces/' + filename)
        surface_obj.readlines(1) # first line is a header

        print('patient', i)

        for j, line in enumerate(surface_obj.readlines()):
            coords = line.split(" ")
            print('xyz coordinates', coords)
            if len(coords) != 3:
                break
            f['surfacespoints'][i, j, :] = [float(coords[0]) + 72.5, float(coords[1]) + 126.5,
                                            float(coords[2]) + 90.5]

        surface_obj.close()

    print("Reading QC labels...")
    label_file = open(os.path.join(path, label_file))
    lines = label_file.readlines()

    for i, line in enumerate(lines[1:]):   # skip header
        patient_id = line.split('+')[1].split('_')[0]
        label = int(line.split(',')[1])

        print('patient:', patient_id, 'label:', label, 'line: ', line)

        if not 'followup' in line and not 'anat_2' in line and not 'anat_3' in line and not 'anat_4' in line and not 'anat_5' in line and not 'anat_6' in line and not 'baseline' in line:
            patient_data[patient_id]['label'] = label

    print("Computing surface distances... Could take a while")

    # compute surface distance volumes in same space as T1
    p = Pool(cores)
    surf_points = np.zeros((40962*2, 3), dtype='float32')

    for i in range(total_subjects):
        surf_points = f['surfacepoints'][i, :, :]

        # p.apply_async(distance_to_surf, args=(surf_points, i,))
        distance_to_surf(surf_points, i)
        print("Launched job", i)

    p.close()
    p.join()

    print("Done ", total_subjects, 'surfaces')

    f.close()

    return 0

def distance_to_surf(surface_points, patient_id):
    surface_distance = np.ones((181, 217, 181), dtype='float32')

    print("surface points: ", np.shape(surface_points))

    floatX = np.zeros(np.shape(surface_distance)[0], dtype='float32')
    floatY = np.zeros(np.shape(surface_distance)[1], dtype='float32')
    floatZ = np.zeros(np.shape(surface_distance)[2], dtype='float32')

    for xx in range(np.shape(floatX)[0]):
        floatX[xx] = float(xx)

    for yy in range(np.shape(floatY)[0]):
        floatY[yy] = float(yy)

    for zz in range(np.shape(floatZ)[0]):
        floatZ[zz] = float(zz)

    print("building KDTree...")
    tree = KDTree(surface_points, leaf_size=10000)
    print("built KDTree!")

    for z in range(np.shape(surface_distance)[0]):
        print("z: ", z)
        for y in range(np.shape(surface_distance)[1]):
            for x in range(np.shape(surface_distance)[2]):
                (distance, index) = tree.query(np.reshape([floatZ[z], floatY[y], floatX[x]], (1, 3)), return_distance = True)
                surface_distance[z, y, x] = distance
                # brute force method, very slow
                # for point in surf_points:
                #     d = euclidean([floatZ[z], floatY[y], floatX[x]], point)

                #     if surface_distance[z,y,x] > d:
                #         surface_distance[z,y,x] = d

    output_filename = os.path.join(output_path, str(patient_id) + '_surface_distance.nii.gz')
    img = nib.Nifti1Image(surface_distance, np.eye(4))

    nib.save(img, output_filename)
    return surface_distance, output_filename


def combine_objs(obj1, obj2, newname):
    print(obj1)
    print(obj2)
    print(newname)
    subprocess.Popen(['objconcat', obj1, obj2, 'none', 'none', newname, 'none'])


def register_MINC(moving_image, atlas, output_image):
    register_command_line = ['/data1/users/adoyle/quarantines/Linux-x86_64/SRC/civet-2.1.0/progs/bestlinreg.pl',
                             '-lsq12',
                             '-nmi',
                             moving_image,
                             atlas,
                             '/tmp/transformation' + str(np.random.randn(1000000)) + '.xfm',
                             output_image,
                             '-clobber',
                             '-target_mask',
                             '/data1/users/adoyle/mni_icbm152_t1_tal_nlin_asym_09a_mask.mnc']

    # subprocess.run(register_command_line, stdout=open(os.devnull, 'wb'))
    subprocess.run(register_command_line)

    return

def register_ants(moving_image, atlas, output_image):
    reg = Registration()

    reg.inputs.fixed_image = atlas
    reg.inputs.moving_image = moving_image
    reg.inputs.output_transform_prefix = 'transform'
    reg.inputs.output_warped_image = output_image
    reg.inputs.output_transform_prefix = "stx-152"
    reg.inputs.transforms = ['Translation']
    reg.inputs.transform_parameters = [(0.1,)]
    reg.inputs.number_of_iterations = ([[10000, 111110, 11110]])
    reg.inputs.dimension = 3
    reg.inputs.write_composite_transform = True
    reg.inputs.collapse_output_transforms = False
    reg.inputs.metric = ['Mattes']
    reg.inputs.metric_weight = [1]
    reg.inputs.radius_or_number_of_bins = [32]
    reg.inputs.sampling_strategy = ['Regular']
    reg.inputs.sampling_percentage = [0.3]
    reg.inputs.convergence_threshold = [1.e-6]
    reg.inputs.convergence_window_size = [20]
    reg.inputs.smoothing_sigmas = [[4, 2, 1]]
    reg.inputs.sigma_units = ['vox']
    reg.inputs.shrink_factors = [[32, 16, 4]]
    reg.inputs.use_estimate_learning_rate_once = [True]
    reg.inputs.use_histogram_matching = [False]
    reg.inputs.initial_moving_transform_com = True

    reg.run()

def check_datasets():

    histograms = {}

    for site in mri_sites:
        histograms[site] = np.zeros(256, dtype='float32')

    bins = np.linspace(0.0, 1.0, 257)

    images = f['MRI']
    datasets = f['dataset']
    filenames = f['filename']

    for i, (img, dataset, filename) in enumerate(zip(images, datasets, filenames)):
        dataset = dataset.decode('UTF-8')
        filename = filename.decode('UTF-8')
        img = np.asarray(img, dtype='float32')

        try:
            histo = np.histogram(img, bins=bins)
            histograms[dataset] += histo[0]
            print(filename, dataset, np.mean(histo[0]), np.var(histo[0]))
        except:
            print('Error for', filename, 'in dataset', dataset)

    fig, axes = plt.subplots(len(mri_sites), 1, sharex=True, figsize=(4, 24))

    for i, site in enumerate(mri_sites):
        try:
            histograms[site] = np.divide(histograms[site], np.sum(histograms[site]))
            axes[i].plot(bins[:-1], histograms[site], lw=2, label=site)

            # axes[i].set_xlim([0, 1])

            axes[i].set_ylabel(site, fontsize=16)
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
        except:
            print('Problem normalizing histogram for', site)

    plt.tight_layout()
    plt.savefig(output_dir + 'dataset_histograms.png', bbox_inches='tight')

if __name__ == "__main__":
    os.environ["LD_LIBRARY_PATH"] = "/home/users/adoyle/quarantines/Linux-x86_64/lib"

    # path = '/data1/data/ABIDE/'

    # for filename in os.listdir(path + '/surfaces/'):
    #     if "right" in filename:
    #         # patient_id = filename.split('+')[1]
    #
    #         filename1 = filename
    #         filename2 = filename.replace("right", "left")
    #         filename3 = filename.replace("right", "combined")
    #
    #         combine_objs(os.path.join(path, 'surfaces/' + filename1), os.path.join(path, 'surfaces/' + filename2), os.path.join(path, 'surfaces/' + filename3))

    # for filename in os.listdir(path + '/T1s/'):
    #     try:
    #         patient_id = filename.split('+')[1]
    #
    #         p = subprocess.Popen(['mincresample', '-nearest_neighbour', '-like', path + 'icbm_template_1.00mm.mnc', path + 'T1s/' + filename, path + 'T1_downsampled/' + patient_id + '.mnc', '-clobber'])
    #         p.communicate()
    #     except:
    #         print filename

    n_abide = count_abide(data_dir + '/deep_abide/', 't1_qc.csv')
    # n_ibis = count_ibis(data_dir + '/IBIS/', 'ibis_t1_qc.csv')
    # n_ping = count_ping(data_dir + '/PING/', 't1_qc.csv')
    n_ds030 = count_ds030(data_dir + '/ds030/', 'ds030_DB.csv')

    print('Subjects with QC labels:')
    print('ABIDE:', n_abide)
    # print('IBIS:', n_ibis)
    # print('PING:', n_ping)
    print('ds030:', n_ds030)

    # total_subjects = n_abide + n_ibis + n_ping + n_ds030

    abide_output = output_dir + 'abide.hdf5'
    ds030_output = output_dir + 'ds030.hdf5'

    dt = h5py.special_dtype(vlen=bytes)

    with h5py.File(abide_output, 'w') as f:
        f.create_dataset('MRI', (n_abide, 1, target_size[0], target_size[1], target_size[2]), dtype='float32')
        f.create_dataset('qc_label', (n_abide,), dtype='float32')

        f.create_dataset('filename', (n_abide,), dtype=dt)
        f.create_dataset('dataset', (n_abide,), dtype=dt)

        next_index = make_abide(data_dir + '/deep_abide/', f, 't1_qc.csv', 0)

        abide_indices = list(range(0, next_index))
        pickle.dump(abide_indices, open(output_dir + 'abide_indices.pkl', 'wb'))

    with h5py.File(ds030_output, 'w') as f:
        f.create_dataset('MRI', (n_ds030, 1, target_size[0], target_size[1], target_size[2]), dtype='float32')
        f.create_dataset('qc_label', (n_ds030,), dtype='float32')

        f.create_dataset('filename', (n_ds030,), dtype=dt)
        f.create_dataset('dataset', (n_ds030,), dtype=dt)

        next_index = make_ds030(data_dir + '/ds030/', f, 'ds030_DB.csv', 0)

        ds030_indices = list(range(0, next_index))
        pickle.dump(ds030_indices, open(output_dir + 'ds030_indices.pkl', 'wb'))

        # print('Starting PING...')
        # next_index = make_ping(data_dir + '/PING/', f, 't1_qc.csv', 0)
        # ping_indices = range(0, next_index)
        # print('Last PING index:', next_index - 1)
        # print('Starting ABIDE...')
        # abide_indices = range(ping_indices[-1] + 1, next_index)
        # print('Last ABIDE index:', next_index - 1)
        # print('Starting IBIS...')
        # next_index = make_ibis(data_dir + '/IBIS/', f, 'ibis_t1_qc.csv', next_index)
        # ibis_indices = range(abide_indices[-1] + 1, next_index)
        # print('Last IBIS index:', next_index - 1)
        # print('Starting ds030...')
        # ds030_indices = range(ibis_indices[-1] + 1, next_index)
        # print('Last ds030 index:', next_index - 1)
        #
        # pickle.dump(ping_indices, open(output_dir + 'ping_indices.pkl', 'wb'))
        # pickle.dump(ibis_indices, open(output_dir + 'ibis_indices.pkl', 'wb'))
        #
        #
        # check_datasets()
