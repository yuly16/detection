from __future__ import print_function, division, absolute_import
import csv
import os
import os.path
import tarfile
from six.moves.urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from experiment import util

object_categories = ['aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor']

urls = {
    'devkit': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar',
    'trainval_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',
    'test_images_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',
    'test_anno_2007': 'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar',
}


def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
            # data.append([name, label])
            # print('%s  %d' % (name, label))
    return data


def read_object_labels(root, mode):
    anno_file = os.path.join(root,'annotations.txt')
    labeled_data = dict()
    f = open(anno_file, 'r')
    anno_list = f.readlines()
    f.close()
    for anno_item in anno_list:
        anno_item = anno_item.strip()
        if mode == 'train':
            if anno_item.split('_')[0] in ['2009','2010','2011','2012']:
                onehot = np.zeros(20)
                for i in anno_item.split(' ')[1:]:
                    onehot[int(i)] = 1
                labeled_data[anno_item.split(' ')[0]] = onehot
            else:
                continue
        elif mode == 'test':
            if anno_item.split('_')[0] in ['2007','2008']:
                onehot = np.zeros(20)
                for i in anno_item.split(' ')[1:]:
                    onehot[int(i)] = 1
                labeled_data[anno_item.split(' ')[0]] = onehot
            else:
                continue
    return labeled_data

def write_object_labels_csv(file, labeled_data):
    # write a csv file
    print('[dataset] write file %s' % file)
    with open(file, 'w',newline='') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(20):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

    csvfile.close()


def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images


def find_images_classification(root, dataset, set):
    path_labels = os.path.join(root, 'VOCdevkit', dataset, 'ImageSets', 'Main')
    images = []
    file = os.path.join(path_labels, set + '.txt')
    with open(file, 'r') as f:
        for line in f:
            images.append(line)
    return images


def download_voc2007(root):
    path_devkit = os.path.join(root, 'VOCdevkit')
    path_images = os.path.join(root, 'VOCdevkit', 'VOC2007', 'JPEGImages')
    tmpdir = os.path.join(root, 'tmp')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(path_devkit):

        if not os.path.exists(tmpdir):
            os.makedirs(tmpdir)

        parts = urlparse(urls['devkit'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['devkit'], cached_file))
            utils.download_url(urls['devkit'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # train/val images/annotations
    if not os.path.exists(path_images):

        # download train/val images/annotations
        parts = urlparse(urls['trainval_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['trainval_2007'], cached_file))
            utils.download_url(urls['trainval_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test annotations
    test_anno = os.path.join(path_devkit, 'VOC2007/ImageSets/Main/aeroplane_test.txt')
    if not os.path.exists(test_anno):

        # download test annotations
        parts = urlparse(urls['test_images_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_images_2007'], cached_file))
            utils.download_url(urls['test_images_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    # test images
    test_image = os.path.join(path_devkit, 'VOC2007/JPEGImages/000001.jpg')
    if not os.path.exists(test_image):

        # download test images
        parts = urlparse(urls['test_anno_2007'])
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(tmpdir, filename)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(urls['test_anno_2007'], cached_file))
            utils.download_url(urls['test_anno_2007'], cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')


class Voc2012Classification(data.Dataset):

    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        # self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        # download dataset
        # download_voc2007(self.root)

        # define path of csv file

        file_csv = os.path.join(self.root, self.set + '.csv')
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            # generate csv file
            labeled_data = read_object_labels(self.root, self.set)
            # write csv file
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] VOC 2012 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        target[target == 0] = -1
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        # img = img.resize((299,299))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
