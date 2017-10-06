import h5py

import matplotlib.pyplot as plt



workdir = '/data1/data/deepqc/'

if __name__ == '__main__':

    f = h5py.File(workdir + 'deepqc.hdf5')

    images = f['MRI']

    for i, image in enumerate(images):
        plt.imshow(image[96, ...])
        plt.savefig(workdir + str(i) + '-' + slice + '.png')