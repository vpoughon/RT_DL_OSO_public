# coding: utf-8

import os
import argparse
import glob
import random
import sys
import numpy as np
import gdal
import gdalconst
from osgeo import ogr
import osgeo.gdal_array


class CnesGenerator:
    patch_size = 512
    validation_patch_size = 0
    validation_patch_stride = 0
    validation_patch_border = 0
    num_bands = 0
    num_samples = 0
    class_id = 0
    class_id_set = []
    gt_cover_pct = 0

    def __load_tif_to_nparray(self, tif_path):
        o_image = gdal.Open(tif_path)
        if o_image is None:
            raise Exception("couldn't open input dataset" + tif_path)

        o_proj = o_image.GetProjection()
        o_geo = o_image.GetGeoTransform()
        t_img = o_image.ReadAsArray()
        return t_img

    def __init__(self, img_path, indices_path, train_labels_path, test_labels_path):
        self.img_path = img_path
        self.indices_path = indices_path
        self.train_labels_path = train_labels_path
        self.test_labels_path = test_labels_path

        self.indices_suffix = '_NDVI_NDWI'
        self.label_suffix = '.tif_label'

        t_img = glob.glob(os.path.join(img_path, '*.tif'))
        if len(t_img) == 0:
            raise Exception('Error, no image found in {}'.format(img_path))

        self.all_tiles = {}
        self.all_gt_train = {}
        self.all_gt_test = {}
        self.all_gt_train_valid_pos = {}

        for img_file in t_img:
            basename, ext = os.path.splitext(img_file)
            basepath, basename = os.path.split(basename)

            file_parts = basename.split("_")
            img_date = file_parts[0]
            tile_id = file_parts[1]

            img_with_indices_file = os.path.join(indices_path, basename + self.indices_suffix + ".tif")
            train_labels_file = os.path.join(train_labels_path, basename + self.label_suffix + ".tif")
            test_labels_file = os.path.join(test_labels_path, basename + self.label_suffix + ".tif")

            #print("Image {0}, image+indices {1}, train labels from {2}, test labels from {3}".format(
            #    img_file, img_with_indices_file, train_labels_file, test_labels_file))

            t_img = self.__load_tif_to_nparray(img_with_indices_file)
            if self.num_bands == 0:
                self.num_bands = t_img.shape[0]
            elif self.num_bands != t_img.shape[0]:
                raise Exception("All tiles must have the same number of bands! Tile {0} does not have {1} bands".format(
                                img_with_indices_file, self.num_bands))

            if tile_id not in self.all_gt_train:
                t_train_labels = self.__load_tif_to_nparray(train_labels_file)
                self.all_gt_train[tile_id] = t_train_labels
                self.all_gt_train_valid_pos[tile_id] = [] # will be initialized later

            if tile_id not in self.all_gt_test:
                t_test_labels = self.__load_tif_to_nparray(test_labels_file)
                self.all_gt_test[tile_id] = t_test_labels

            #print("Image size: {0}, labels size: {1}".format(t_img.shape, t_train_labels.shape))
            if tile_id not in self.all_tiles:
                self.all_tiles[tile_id] = []
            self.all_tiles[tile_id].append(t_img)

        for tile_id in self.all_tiles:
            num_samples = len(self.all_tiles[tile_id])
            if self.num_samples == 0:
                self.num_samples = num_samples
            elif self.num_samples != num_samples:
                raise Exception("All tiles must have the same number of time samples! Tile {0} does not have {1} samples".format(
                                tile_id, self.num_samples))

            tile_size = self.all_tiles[tile_id][0].shape

            tiles_concat = np.zeros((self.num_bands * num_samples, tile_size[1], tile_size[2]), dtype=np.float32)

            for k in range(num_samples):
                tiles_concat[(k * self.num_bands):((k+1) * self.num_bands), :, :] = self.all_tiles[tile_id][k]

            self.all_tiles[tile_id] = tiles_concat
            print("concatenated all samples for {0}, shape is {1}".format(tile_id, tiles_concat.shape))
            o_image = None

    def compute_all_valid_positions(self):
        import cv2
        if len(self.class_id_set) == 0:
            raise Exception("Please provide a valid class id set first")
        if self.gt_cover_pct == 0:
            raise Exception("Please provide a valid gt cover pct first")

        for tile_id in self.all_gt_train_valid_pos:
            # for each class, get integral image
            valid_pos_in_class = {}
            for class_id in self.class_id_set:
                mask = (self.all_gt_train[tile_id] == class_id).astype("uint8")
                integral_mask = cv2.integral(mask)
                ps = self.patch_size
                sum_mask = (integral_mask[0:-ps,0:-ps] + integral_mask[ps:,ps:]
                                                       - integral_mask[0:-ps,ps:] - integral_mask[ps:,0:-ps])
                positive_positions = np.where(sum_mask > self.gt_cover_pct * ps * ps)
                valid_pos_in_class[class_id] = positive_positions # store top left corner of the valid patch

            self.all_gt_train_valid_pos[tile_id] = valid_pos_in_class


    def get_all_pixel_data_and_class(self, use_test_data_set):
        if len(self.class_id_set) == 0:
            raise Exception("Please provide a valid class id set first")

        label_map_idx = {}
        idx = 0
        for lbl in self.class_id_set:
            label_map_idx[lbl] = idx
            idx += 1

        if use_test_data_set:
            label_tiles = self.all_gt_test
        else:
            label_tiles = self.all_gt_train

        all_train_data = None
        all_train_labels = None
        for tile_id in self.all_tiles:
            all_train_data_tile = np.zeros((0, self.all_tiles[tile_id].shape[0]), np.float32)
            all_train_labels_tile = np.zeros((0, 1), np.float32)
            # for each class, get integral image
            for class_id in self.class_id_set:
                valid_positions = np.where(label_tiles[tile_id] == class_id)
#                print (valid_positions[0].shape, valid_positions[1].shape)
                class_pixel_data = self.all_tiles[tile_id][:, valid_positions[0], valid_positions[1]]
#                print(class_pixel_data.shape)
                all_train_data_tile = np.vstack((all_train_data_tile, np.transpose(class_pixel_data)))
                all_train_labels_tile = np.vstack((all_train_labels_tile, np.ones((class_pixel_data.shape[1], 1), np.float32)*label_map_idx[class_id]))

            if all_train_data is None:
                all_train_data = all_train_data_tile
                all_train_labels = all_train_labels_tile
            else:
                all_train_data = np.vstack((all_train_data, all_train_data_tile))
                all_train_labels = np.vstack((all_train_labels, all_train_labels_tile))

        return all_train_data, all_train_labels

        #TODO: make one-hot

    def get_all_pixel_data_unlabeled(self):
        if len(self.class_id_set) == 0:
            raise Exception("Please provide a valid class id set first")

        all_unlabeled_data = None
        for tile_id in self.all_tiles:
            all_train_data_tile = np.zeros((0, self.all_tiles[tile_id].shape[0]), np.float32)
            all_train_labels_tile = np.zeros((0, 1), np.float32)
            # for each class, get integral image

            valid_positions = np.where(np.logical_and(self.all_gt_test[tile_id] == 0, self.all_gt_train[tile_id] == 0))

            unlabeled_pixel_data = np.transpose(self.all_tiles[tile_id][:, valid_positions[0], valid_positions[1]])
            if all_unlabeled_data is None:
                all_unlabeled_data = unlabeled_pixel_data
            else:
                all_unlabeled_data = np.vstack((all_unlabeled_data, unlabeled_pixel_data))

        return all_unlabeled_data

    def set_patch_size(self, patch_size):
        self.patch_size = patch_size

    def set_validation_patch_sampling(self, patch_size, patch_stride, patch_border):
        self.validation_patch_size = patch_size
        self.validation_patch_stride = patch_stride
        self.validation_patch_border = patch_border #to avoid Unet issues with borders

    def get_num_bands(self):
        return self.num_bands

    def get_num_samples(self):
        return self.num_samples

    def set_single_class(self, class_id):
        class_ids = self.get_class_ids()
        if class_id not in list(class_ids):
            raise Exception("Cannot set target class to {0}, it does not appear in the dataset, valid classes are {1}".format(class_id, class_ids))
        self.class_id = class_id

    def set_multi_class(self, class_id_set):
        class_ids = self.get_class_ids()
        for class_id in class_id_set:
            if class_id not in list(class_ids):
                raise Exception("Cannot set target class to {0}, it does not appear in the dataset, valid classes are {1}".format(class_id, class_ids))
        self.class_id_set = class_id_set

    def set_min_gt_cover(self, gt_cover_pct):
        if gt_cover_pct < 0 or gt_cover_pct > 1:
            raise Exception("Ground truth cover percentage must be a value between 0 and 1")
        self.gt_cover_pct = gt_cover_pct

    def get_class_ids(self):
        all_class_ids = np.zeros((0,), dtype=np.int32)
        for tile_id in self.all_tiles:
            classes_tile = np.unique(self.all_gt_train[tile_id]).astype(np.int32)
            all_class_ids = np.union1d(all_class_ids, classes_tile)
        return all_class_ids

    def get_class_stats_train(self):
        all_class_ids = list(self.get_class_ids())
        class_stats = {}
        #TODO: handle multiple tiles
        for tile_id in self.all_tiles:
            for cid in all_class_ids:
                class_stats[cid] = np.sum(self.all_gt_train[tile_id]==cid)
        return class_stats

    def __clip_patch_dim_to_bounds(self, v, maxv, patch_size):
        if v < 0:
            ty = 0
            dy = -v
            ny = patch_size + v
        elif v > maxv - patch_size:
            ty = v
            dy = 0
            ny = v - (maxv - patch_size)
        else:
            ty = v
            dy = 0
            ny = patch_size
        return ty, dy, ny

    def get_validation_patches(self):
        if self.validation_patch_size == 0 or self.validation_patch_stride == 0:
            raise Exception("Validation patch size or stride not set, call set_validation_patch_sampling first")

        for tile_id in self.all_tiles:
            start_y = -self.validation_patch_border
            end_y = self.all_tiles[tile_id].shape[1] + self.validation_patch_border - self.validation_patch_size

            start_x = -self.validation_patch_border
            end_x = self.all_tiles[tile_id].shape[2] + self.validation_patch_border - self.validation_patch_size

            range_y = range(start_y, end_y + self.validation_patch_stride, self.validation_patch_stride)
            range_x = range(start_x, end_x + self.validation_patch_stride, self.validation_patch_stride)

            total_valid_patches = len(range_x) * len(range_y)

            tile_patches_validation = np.zeros((total_valid_patches, 
                                                self.num_bands * self.num_samples, 
                                                self.validation_patch_size, 
                                                self.validation_patch_size), dtype=np.float32)
            tile_gt_validation = np.zeros((total_valid_patches, len(self.class_id_set) + 1, 
                                           self.validation_patch_size, 
                                           self.validation_patch_size), dtype=self.all_gt_test[tile_id].dtype)
            idx = 0
            for y in range_y:
                for x in range_x:
                    ty, dy, ny = self.__clip_patch_dim_to_bounds(y, self.all_tiles[tile_id].shape[1], self.validation_patch_size)
                    tx, dx, nx = self.__clip_patch_dim_to_bounds(x, self.all_tiles[tile_id].shape[2], self.validation_patch_size)

                    #print("Extracting patch at {},{}: copy {} x {} values from {},{} to {},{}".format(y,x,ny,nx, ty,tx, dy,dx))
                    tile_patches_validation[idx, :, dy:(dy+ny), dx:(dx+nx)] = self.all_tiles[tile_id][:, ty:(ty+ny), tx:(tx+nx)]
                    gt_test_patch = self.all_gt_test[tile_id][ty:(ty+ny), tx:(tx+nx)]
                    onehot_gt_test_patch = np.zeros((len(self.class_id_set) + 1, gt_test_patch.shape[0], gt_test_patch.shape[1]))
                    for ch in range(len(self.class_id_set)):
                        onehot_gt_test_patch[ch,...] = gt_test_patch == self.class_id_set[ch]
                   
                    tile_gt_validation[idx, :, dy:(dy+ny), dx:(dx+nx)] = onehot_gt_test_patch
                    idx += 1
                    #TODO: mirror borders where necessary
        return tile_patches_validation, tile_gt_validation

    def generate_train_patch_multiclass(self, class_id):
        if len(self.class_id_set) == 0:
            raise Exception("Set the desired classes first with set_multi_class")

        done = False
        img_patch = None
        gt_patch = None

        while not done:
            rand_tile_id = random.choice(list(self.all_tiles))

            positive_positions = np.where(self.all_gt_train[rand_tile_id] == class_id)
            idx_pos_rand = np.random.randint(0, positive_positions[0].shape[0])
            rand_delta_x = np.random.randint(-self.patch_size // 3, self.patch_size // 3)
            rand_delta_y = np.random.randint(-self.patch_size // 3, self.patch_size // 3)

            tile_size = self.all_tiles[rand_tile_id].shape

            r_left = np.clip(positive_positions[0][idx_pos_rand] + rand_delta_x, 0, tile_size[1] - self.patch_size - 1)
            r_top = np.clip(positive_positions[1][idx_pos_rand] + rand_delta_y, 0, tile_size[2] - self.patch_size - 1)

            img_patch = self.all_tiles[rand_tile_id][:, r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]
            gt_patch = np.zeros((len(self.class_id_set) + 1, img_patch.shape[1], img_patch.shape[2]))
            i_in_set = -1
            for i in range(len(self.class_id_set)):
                if self.class_id_set[i] == class_id:
                    i_in_set = i
                gt_patch[i,...] = self.all_gt_train[rand_tile_id][r_left:(r_left+self.patch_size),
                                                                  r_top:(r_top + self.patch_size)] == self.class_id_set[i]
                gt_patch[len(self.class_id_set),...] += gt_patch[i,...]
            gt_patch[len(self.class_id_set),...] = 1 - gt_patch[len(self.class_id_set),...]

            if i_in_set < 0:
                raise Exception("Class id must be part of the class id set !")

             # randomly transpose, flip up or down
            img_patch = np.transpose(img_patch,(1,2,0))
            gt_patch = np.transpose(gt_patch,(1,2,0))
            if (random.randint(0, 1) == 1):
                img_patch = np.transpose(img_patch, (0, 2, 1))
                gt_patch = np.transpose(gt_patch, (0, 2, 1))
            if (random.randint(0, 1) == 1):
                img_patch = np.fliplr(img_patch)
                gt_patch = np.fliplr(gt_patch)
            if (random.randint(0, 1) == 1):
                img_patch = np.flipud(img_patch)
                gt_patch = np.flipud(gt_patch)
            img_patch = np.transpose(img_patch,(2,0,1))
            gt_patch = np.transpose(gt_patch,(2,0,1))

            if np.sum(gt_patch[i_in_set,...]) / (float(self.patch_size) * float(self.patch_size)) > self.gt_cover_pct:
                done = True
            #else:
            #    print(np.sum(gt_patch[i_in_set,...]) )


        return img_patch, gt_patch

    def generate_train_patch_fast(self, class_id, nb_patch):
        if len(self.class_id_set) == 0:
            raise Exception("Set the desired classes first with set_multi_class")

        img_patch = None
        gt_patch = None

        rand_tile_id = random.choice(list(self.all_tiles))
        if self.all_gt_train_valid_pos[rand_tile_id] is None:
            raise Exception("No valid positions were pre-computed, call compute_all_valid_positions() first")

        positive_positions = self.all_gt_train_valid_pos[rand_tile_id][class_id]
        idx_pos_rand = np.random.randint(0, positive_positions[0].shape[0])

        r_left = positive_positions[0][idx_pos_rand]
        r_top = positive_positions[1][idx_pos_rand]
        img_patch = self.all_tiles[rand_tile_id][:, r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]
        gt_patch = np.zeros((len(self.class_id_set) + 1, img_patch.shape[1], img_patch.shape[2]))
        i_in_set = -1
        for i in range(len(self.class_id_set)):
            if self.class_id_set[i] == class_id:
                i_in_set = i
            gt_patch[i,...] = self.all_gt_train[rand_tile_id][r_left:(r_left+self.patch_size),
                                                              r_top:(r_top + self.patch_size)] == self.class_id_set[i]
            gt_patch[len(self.class_id_set),...] += gt_patch[i,...]
        gt_patch[len(self.class_id_set),...] = 1 - gt_patch[len(self.class_id_set),...]

        if i_in_set < 0:
            raise Exception("Class id must be part of the class id set !")

        # randomly transpose, flip up or down
        img_patch = np.transpose(img_patch,(1,2,0))
        gt_patch = np.transpose(gt_patch,(1,2,0))
        if (random.randint(0, 1) == 1):
            img_patch = np.transpose(img_patch, (1, 0, 2))
            gt_patch = np.transpose(gt_patch, (1, 0, 2))
        if (random.randint(0, 1) == 1):
            img_patch = np.fliplr(img_patch)
            gt_patch = np.fliplr(gt_patch)
        if (random.randint(0, 1) == 1):
            img_patch = np.flipud(img_patch)
            gt_patch = np.flipud(gt_patch)
        img_patch = np.transpose(img_patch,(2,0,1))
        gt_patch = np.transpose(gt_patch,(2,0,1))

        return img_patch, gt_patch


    def generate_train_patch(self):
        if self.class_id == 0:
            raise Exception("Set the desired class first with set_single_class")

        done = False
        img_patch = None
        gt_patch = None

        while not done:
            rand_tile_id = random.choice(list(self.all_tiles))

            positive_positions = np.where(self.all_gt_train[rand_tile_id] == self.class_id)
            idx_pos_rand = np.random.randint(0, positive_positions[0].shape[0])
            rand_delta_x = np.random.randint(-self.patch_size // 3, self.patch_size // 3)
            rand_delta_y = np.random.randint(-self.patch_size // 3, self.patch_size // 3)

            tile_size = self.all_tiles[rand_tile_id].shape
#            r_left = random.randint(0, tile_size[1] - self.patch_size - 1)
#            r_top = random.randint(0, tile_size[2] - self.patch_size - 1)

            r_left = np.clip(positive_positions[0][idx_pos_rand] + rand_delta_x, 0, tile_size[1] - self.patch_size - 1)
            r_top = np.clip(positive_positions[1][idx_pos_rand] + rand_delta_y, 0, tile_size[2] - self.patch_size - 1)

            img_patch = self.all_tiles[rand_tile_id][:, r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]
            gt_patch = self.all_gt_train[rand_tile_id][r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]==self.class_id

            if np.sum(gt_patch) / (float(self.patch_size) * float(self.patch_size)) > self.gt_cover_pct:
                done = True


        return img_patch, gt_patch

    def get_tile_sample_band(self, patch, sample_id, band_id):
        return None
