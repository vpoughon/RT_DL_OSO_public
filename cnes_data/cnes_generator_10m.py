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
#import cv2
import pickle

class CnesGeneratorSentinel:
    patch_size = 512
    validation_patch_size = 0
    validation_patch_stride = 0
    validation_patch_border = 0
    num_bands = 0
    num_samples = 0
    class_id_set = []
    class_id_idx = {}
    gt_cover_pct = 0
    dense_annotations = None
    running_stats = None
    add_contours_channel = False
    compute_running_stats = False

    single_scheduler_mode = False
    full_tile_width = 0
    full_tile_height = 0

    def __compute_subtile_size_and_position(self, rows, cols, num_split_tiles, split_id):
        split_modes = [[1, 1], [2, 1], [3, 1], [2, 2], [5, 1], [3, 2], [7, 1], [4, 2],
                       [3, 3], [5, 2], [11, 1], [4, 3], [13, 1], [7, 2], [5, 3], [4, 4]]

        num_split_tiles = num_split_tiles - 1
        if num_split_tiles > len(split_modes):
            raise Exception("Too many splits, split mode not yet implemented")

        rows_tile = rows // split_modes[num_split_tiles][0]
        cols_tile = cols // split_modes[num_split_tiles][1]

        index_row = split_id // split_modes[num_split_tiles][1]
        index_col = split_id % split_modes[num_split_tiles][1]

        return split_modes[num_split_tiles][0], split_modes[num_split_tiles][1], rows_tile, cols_tile, index_row, index_col

    def __get_tile_id_for_position(self, x, y):
        for tile_id in range(self.num_split_tiles):
            _, _, rows_tile, cols_tile, index_row, index_col = self.__compute_subtile_size_and_position(self.full_tile_height,
                                                                                                        self.full_tile_width,
                                                                                                        self.num_split_tiles,
                                                                                                        tile_id)

            tile_rectangle = (rows_tile * index_row, cols_tile * index_col, rows_tile * index_row+rows_tile, cols_tile * index_col+cols_tile)
#            print(tile_id, tile_rectangle)
            if x >= tile_rectangle[1] and x < tile_rectangle[3] and y >= tile_rectangle[0] and y < tile_rectangle[2]:
                return tile_id, tile_rectangle

        raise Exception("Trying to find subtile _id for invalid x={},y={}".format(x, y))

    def __initialize_full_tile_info(self, tif_path):
        o_image = gdal.Open(tif_path)
        if o_image is None:
            raise Exception("couldn't open input dataset" + tif_path)

        #tile shape: C x H x W

        self.full_tile_width = o_image.RasterXSize
        self.full_tile_height = o_image.RasterYSize
#        bands = o_image.RasterCount

    def __load_tif_to_nparray(self, tif_path, num_split_tiles = 1, split_id = 0, debug_init_zeros=False):
        o_image = gdal.Open(tif_path)
        if o_image is None:
            raise Exception("couldn't open input dataset" + tif_path)

        o_proj = o_image.GetProjection()
        o_geo = o_image.GetGeoTransform()

        cols = o_image.RasterXSize
        rows = o_image.RasterYSize
        bands = o_image.RasterCount

        n_tiles_x, n_tiles_y, rows_tile, cols_tile, index_row, index_col = self.__compute_subtile_size_and_position(rows, cols, num_split_tiles, split_id)

        #t_img = o_image.ReadAsArray()

        print("Loading tile, split into {} x {} subtiles, tile id is {} giving subtile ({}, {}) of size {}x{}".format(
            n_tiles_x, n_tiles_y,
            split_id, rows_tile * index_row, cols_tile * index_col,
            rows_tile, cols_tile))
        sys.stdout.flush()

        if debug_init_zeros:
            print("Generating debug 0-image of size {}".format((bands, rows_tile, cols_tile)))
            if bands == 1:
                if num_split_tiles > 1:
                    t_img = np.zeros((rows_tile, cols_tile), np.int16)
                else:
                    t_img = np.zeros((rows, cols), np.int16)
            else:
                if num_split_tiles > 1:
                    t_img = np.zeros((bands, rows_tile, cols_tile), np.int16)
                else:
                    t_img = np.zeros((bands, rows, cols), np.int16)
        else:
            if num_split_tiles > 1:
                # 	doc:: gdal.Dataset.ReadAsArray(self, xoff=0, yoff=0, xsize=None, ysize=None, ...
                t_img = o_image.ReadAsArray(cols_tile * index_col, rows_tile * index_row, cols_tile, rows_tile)
            else:
                t_img = o_image.ReadAsArray()

            if t_img is None:
                raise Exception("GDAL could not read image as array, ds has {} raster bands".format(o_image.RasterCount))

        return t_img, (rows_tile * index_row, cols_tile * index_col, rows_tile, cols_tile)
    
    def __load_valid_positions(self, vald_pos_path, total_number_pixels, valid_rectangle):
        valid_positions_subtile = {}
        with open(vald_pos_path, 'rb') as f:
            all_valid_positions = pickle.load(f)

        tile_class_freq = np.zeros((len(self.class_id_set),), np.float32)
        # patch VP
        #for cid in self.class_id_set:
            #if cid not in all_valid_positions:
                #raise Exception("Class id {} required for training not found in precomputed valid positions data".format(cid))

        #for cid in self.class_id_set:
            #if all_valid_positions[cid][0].size == 0:
                #continue
        for cid in self.class_id_set:
            if cid not in all_valid_positions:
                print("Warning : Class id {} required for training not found in precomputed valid positions data".format(cid))
                continue
            if all_valid_positions[cid][0].size == 0:
                continue

            filter_y = np.logical_and(all_valid_positions[cid][0] >= valid_rectangle[0],
                                      all_valid_positions[cid][0] < valid_rectangle[0] + valid_rectangle[2])

            filter_x = np.logical_and(all_valid_positions[cid][1] >= valid_rectangle[1],
                                      all_valid_positions[cid][1] < valid_rectangle[1] + valid_rectangle[3])

            filter = np.logical_and(filter_x, filter_y)

            #global frequencies, not only subtile
            tile_class_freq[self.class_id_idx[cid]] = float(all_valid_positions[cid][0].size) / total_number_pixels

            if np.any(filter):
                valid_positions_subtile[cid] = (all_valid_positions[cid][0][filter] - valid_rectangle[0],
                                                all_valid_positions[cid][1][filter] - valid_rectangle[1])

                if "CNES_DBG_VALID_POSITIONS" in os.environ:
                    print("for tile: y:{}, x:{}, h:{} x w:{}, class={}:  {} positions kept out of {} = {:.2f}%, positions between {},{} - {},{}".format(
                        valid_rectangle[0], valid_rectangle[1],
                        valid_rectangle[2], valid_rectangle[3],
                        cid, np.sum(filter), all_valid_positions[cid][0].size,
                        100 * np.sum(filter) / float(all_valid_positions[cid][0].size),
                        np.min(valid_positions_subtile[cid][1]),
                        np.min(valid_positions_subtile[cid][0]),
                        np.max(valid_positions_subtile[cid][1]),
                        np.max(valid_positions_subtile[cid][0])))
                    sys.stdout.flush()
            else:
                if "CNES_DBG_VALID_POSITIONS" in os.environ:
                    print("for tile: y:{}, x:{}, h:{} x w:{}, class={}:  {} positions kept out of {} = {:.2f}%".format(
                        valid_rectangle[0], valid_rectangle[1],
                        valid_rectangle[2], valid_rectangle[3],
                        cid, np.sum(filter), all_valid_positions[cid][0].size, 0))
                    sys.stdout.flush()

        return valid_positions_subtile, np.asarray(tile_class_freq)

    def assert_single_file(self, filename_list, filename_path):
        if len(filename_list) == 0:
            raise Exception('Error, no file found in {}'.format(filename_path))
        if len(filename_list) > 1:
            raise Exception('Error, too many files found in {}'.format(filename_path))
            
    def compute_min_max_channels(self):
        min_channels = np.zeros(self.num_bands)
        max_channels = np.zeros(self.num_bands)
        for j in range(self.num_bands):
            min_channels[j] = np.percentile(self.tile[j,...],3).astype(np.float32)
            max_channels[j] = np.percentile(self.tile[j,...],97).astype(np.float32)
        self.min_channels = min_channels
        self.max_channels = max_channels

    def enable_contours_channel(self, enable):
        self.add_contours_channel = enable

    def enable_running_stats(self, enable):
        self.compute_running_stats = enable

    def __init__(self, tile_name, img_path, raster_path, num_split_tiles = 1, split_id = 0, single_scheduler_mode = False, traintest="TRAIN"):
        '''
        @param tile_name : name of the tile (for instance : T30TWT)
        @param img_path : path containing a directory for each tile
        @param raster_path : directory containing rasterized labeled data for each tile
        '''
        # initialize attributes
        self.img_path = img_path
        self.raster_path = raster_path 
        self.tile_name = tile_name
        self.traintest = traintest
        self.single_scheduler_mode = single_scheduler_mode

        # get filenames
        tile_filename = glob.glob(os.path.join(img_path, tile_name + '/*.tif'))
        self.assert_single_file(tile_filename, os.path.join(img_path, tile_name))
        
        train_labels_filename = glob.glob(os.path.join(self.raster_path, tile_name + '*training.tif'))
        self.assert_single_file(train_labels_filename, self.raster_path)

        #min_channels_filename = glob.glob(os.path.join(img_path, tile_name + '/*.tif_min_channels.npy'))
        #self.assert_single_file(min_channels_filename, os.path.join(img_path, tile_name), '/*.tif_min_channels.npy')
        
        #max_channels_filename = glob.glob(os.path.join(img_path, tile_name + '/*.tif_max_channels.npy'))
        #self.assert_single_file(max_channels_filename, os.path.join(img_path, tile_name), '/*.tif_max_channels.npy')
        min_channels_filename = glob.glob(os.path.join(img_path, tile_name, '*_min_channels.npy'))
        self.assert_single_file(min_channels_filename, os.path.join(img_path, tile_name))
        
        max_channels_filename = glob.glob(os.path.join(img_path, tile_name, '*_max_channels.npy'))
        self.assert_single_file(max_channels_filename, os.path.join(img_path, tile_name))

        self.__initialize_full_tile_info(tile_filename[0])

        # load data
        self.tile, self.valid_rectangle = self.__load_tif_to_nparray(tile_filename[0], num_split_tiles, split_id, "CNES_DBG_NO_TILE_LOAD" in os.environ)

        self.gt_train, _ = self.__load_tif_to_nparray(train_labels_filename[0], num_split_tiles, split_id)

        if self.num_bands == 0:
            self.num_bands = self.tile.shape[0]
        elif self.num_bands != self.tile.shape[0]:
            raise Exception("All tiles must have the same number of bands! Tile {0} does not have {1} bands".format(
                tile_name, self.num_bands))

        # get min and max for each channel (based on percentile)
        #self.compute_min_max_channels()
        self.min_channels = np.load(min_channels_filename[0])
        self.max_channels = np.load(max_channels_filename[0])

        self.num_split_tiles = num_split_tiles
        self.split_id = split_id
    
    def set_patch_size(self, patch_size):
        self.patch_size = patch_size
        
    def set_dense_annotations(self, anno_filename):
        self.dense_annotations = self.dense_annotations = self.__load_tif_to_nparray(anno_filename, self.num_split_tiles, self.split_id)
        assert(self.dense_annotations.shape[0] == self.gt_train.shape[0])
        assert(self.dense_annotations.shape[1] == self.gt_train.shape[1])

    def set_validation_patch_sampling(self, patch_size, patch_stride, patch_border):
        self.validation_patch_size = patch_size
        self.validation_patch_stride = patch_stride
        self.validation_patch_border = patch_border #to avoid Unet issues with borders

    def get_tile_size(self):
        return self.tile.shape[1], self.tile.shape[2]

    def get_data_type(self):
        return self.tile.dtype

    def get_num_bands(self):
        return self.num_bands

    def get_num_samples(self):
        return self.num_samples

#    def set_single_class(self, class_id):
#        class_ids = self.get_class_ids()
#        if class_id not in list(class_ids):
#            raise Exception("Cannot set target class to {0}, it does not appear in the dataset, valid classes are {1}".format(class_id, class_ids))
#        self.class_id = class_id

    def set_multi_class(self, class_id_set):
        self.class_id_set = class_id_set
        for k in range(len(self.class_id_set)):
            self.class_id_idx[self.class_id_set[k]] = k

        if self.running_stats is None:
            self.running_stats = {class_id: 0 for class_id in self.class_id_set}
        self.running_stats[0] = 0
    
        #valid_pos_filename = glob.glob(self.img_path + '/RasterData/' + self.tile_name + '*training.tif.pickle')
        valid_pos_filename = glob.glob(os.path.join(self.raster_path, self.tile_name + '*training*.pickle'))
        self.assert_single_file(valid_pos_filename, os.path.join(self.img_path, self.tile_name))

        total_number_pixels = self.tile.shape[1] * self.tile.shape[2]
        print("Total pixels for subtile {} = {} x {} = {}".format(self.split_id, self.tile.shape[1], self.tile.shape[2], total_number_pixels))
        sys.stdout.flush()

        self.gt_train_valid_pos = None
        self.class_freq = None

        if self.traintest != "TEST":
            valid_rectangle = self.valid_rectangle
            if self.single_scheduler_mode:
                valid_rectangle = [0, 0, self.full_tile_height, self.full_tile_width]

            gt_train_valid_pos, class_freq = self.__load_valid_positions(valid_pos_filename[0],
                                                                                   total_number_pixels,
                                                                                   valid_rectangle)
            if not self.single_scheduler_mode or (self.single_scheduler_mode and self.split_id == 0):
                self.gt_train_valid_pos = gt_train_valid_pos
                self.class_freq = class_freq
    
    def compute_valid_positions_multi_tiles(self, t_tiles, u_size_hvd):
        '''
        concatenate valid positions for all input tiles
        '''
        d_valid_pos_all_tiles = dict() # dict, key : classid, value : tuple (list coord x, list coord y, rank on which patch can be found)
        d_rank_tile = dict() # link between rank and tile index in t_tiles. Keys : tile index, values : rank list
        d_tile_rank = dict()
        for u_rank in range(u_size_hvd):
            d_tile_rank[u_rank] = int(u_rank * len(t_tiles) //  u_size_hvd)
        for rank, id_tile in d_tile_rank.items():
            if id_tile not in d_rank_tile.keys():
                d_rank_tile[id_tile] = [rank]
            else:
                d_rank_tile[id_tile].append(rank)
        print(d_rank_tile)
        for u_index, s_tile in enumerate(t_tiles):
            valid_pos_filename = glob.glob(os.path.join(self.raster_path, s_tile + '*training*.pickle'))
            self.assert_single_file(valid_pos_filename, os.path.join(self.img_path, self.tile_name))
            valid_positions_subtile = {}
            with open(valid_pos_filename[0], 'rb') as f:
                d_valid_positions = pickle.load(f)
                for classid, value in d_valid_positions.items():
                    if classid in d_valid_pos_all_tiles.keys():
                        # concatenation and adding the rank (randomly if several) where tile can be found
                        d_valid_pos_all_tiles[classid] = (np.concatenate((d_valid_pos_all_tiles[classid][0], d_valid_positions[classid][0])), \
                                                          np.concatenate((d_valid_pos_all_tiles[classid][1], d_valid_positions[classid][1])), \
                                                          np.concatenate((d_valid_pos_all_tiles[classid][2], np.array([random.choice(d_rank_tile[u_index]) for i in range(len(d_valid_positions[classid][1]))]))))
                    else:
                        d_valid_pos_all_tiles[classid] = (d_valid_positions[classid][0], d_valid_positions[classid][1], \
                                                          np.array([random.choice(d_rank_tile[u_index]) for i in range(len(d_valid_positions[classid][1]))]))
        self.d_valid_pos_all_tiles = d_valid_pos_all_tiles
                
            

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

    def get_class_stats_train(self, nb_patchs, nb_epoch):
        # patch VP : il ne faut pas se baser sur la frequence dans les annotations ou dans les images, mais de celle dans les patchs generes
        # la generation des patchs fait en sorte d'equilibrer les classes, et la frequence des classes dans les patchs n'est pas la meme que dans les annotations
        if self.class_id_set is None:
            raise Exception("Set class label map before trying to get statistics")
        #return self.class_freq #[np.asarray(self.class_id_set)]

        class_stats = np.zeros(len(self.class_id_set))
        #print(class_stats.shape)
        
        #for i in range(nb_patchs):
            #patch, gt_patch = self.generate_train_patch_fast(1)
            #for ct in range(len(self.class_id_set)):
                #positive_positions = np.where(gt_patch[:,ct,...] > 0)
                #class_stats[ct] += len(positive_positions[0])
        for i in range(nb_epoch):
            patch, gt_patch = self.generate_train_patch_fast(nb_patchs)
            for ct in range(len(self.class_id_set)):
                positive_positions = np.where(gt_patch[:,ct,...] > 0)
                class_stats[ct] += len(positive_positions[0])
                
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

    def get_validation_patches(self,r_left,r_top):
        '''
        create a pair of patch/ground truth at a specific location. This is typically used for testing
        @param r_left: the left position of the patch
        @param r_top: the top position of the patch
        @return img_patch: the corresponding image patch of size (1,nb_channels,patch_size,patch_size)
        @return gt_patch: the corresponding ground truth patch of size (1,nb_classes,patch_size,patch_size)
        '''
        batch = 0
        # handle padding if patch is out of
        left_padding = np.max([-r_left,0]).astype("int32")
        top_padding = np.max([-r_top,0]).astype("int32")
        right_padding = np.max([-self.tile.shape[1]+r_left + self.patch_size,0]).astype("int32")
        bottom_padding = np.max([-self.tile.shape[2]+r_top + self.patch_size,0]).astype("int32")
        
        if r_left < 0:
            r_left = 0
        if r_top < 0:
            r_top = 0
            
        img_patch = self.tile[:, r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]
        gt_patch = np.zeros((len(self.class_id_set) + 1, img_patch.shape[1], img_patch.shape[2]))
        i_in_set = -1
        for i in range(len(self.class_id_set)):
            gt_patch[i,...] = self.gt_train[r_left:(r_left+self.patch_size),
                                            r_top:(r_top + self.patch_size)] == self.class_id_set[i]
            gt_patch[len(self.class_id_set),...] += gt_patch[i,...]
            gt_patch[len(self.class_id_set),...] = 1 - gt_patch[len(self.class_id_set),...]
        img_patch = np.expand_dims(img_patch, axis=0)
        gt_patch = np.expand_dims(gt_patch, axis=0)

        # convert to float and normalize
        img_patch = img_patch.astype('float32')
        gt_patch = gt_patch.astype('float32')
        for ch in range(len(self.min_channels)):
            img_patch[:,ch,...] = np.clip(img_patch[:,ch,...],self.min_channels[ch],self.max_channels[ch])
            img_patch[:,ch,...] = (img_patch[:,ch,...]-self.min_channels[ch]) / (self.max_channels[ch]-self.min_channels[ch])
        
        # if validation patch is not of patch_size (typically at border), add padding
        img_patch = np.pad(img_patch,
                           ((0,0),(0,0),
                            (left_padding,right_padding),
                            (top_padding,bottom_padding)),
                           "reflect")
        gt_patch = np.pad(gt_patch,
                          ((0,0),(0,0),
                           (left_padding,right_padding),
                           (top_padding,bottom_padding)), 
                          "reflect")
        img_patch = img_patch[:,:,:self.patch_size,:self.patch_size]
        gt_patch = gt_patch[:,:,:self.patch_size,:self.patch_size]
        return img_patch, gt_patch
    
    def get_validation_patches_without_gt(self,r_left,r_top):
        '''
        create a patch at a specific location. This is typically used for testing
        @param r_left: the left position of the patch
        @param r_top: the top position of the patch
        @return img_patch: the corresponding image patch of size (1,nb_channels,patch_size,patch_size)
        '''
        batch = 0
        # handle padding if patch is out of
        left_padding = np.max([-r_left,0]).astype("int32")
        top_padding = np.max([-r_top,0]).astype("int32")
        right_padding = np.max([-self.tile.shape[1]+r_left + self.patch_size,0]).astype("int32")
        bottom_padding = np.max([-self.tile.shape[2]+r_top + self.patch_size,0]).astype("int32")
        
        if r_left < 0:
            r_left = 0
        if r_top < 0:
            r_top = 0
            
        img_patch = self.tile[:, r_left:(r_left+self.patch_size), r_top:(r_top + self.patch_size)]
        #gt_patch = np.zeros((len(self.class_id_set) + 1, img_patch.shape[1], img_patch.shape[2]))
        i_in_set = -1
        #for i in range(len(self.class_id_set)):
            #gt_patch[i,...] = self.gt_train[r_left:(r_left+self.patch_size),
                                            #r_top:(r_top + self.patch_size)] == self.class_id_set[i]
            #gt_patch[len(self.class_id_set),...] += gt_patch[i,...]
            #gt_patch[len(self.class_id_set),...] = 1 - gt_patch[len(self.class_id_set),...]
        img_patch = np.expand_dims(img_patch, axis=0)
        #gt_patch = np.expand_dims(gt_patch, axis=0)

        # convert to float and normalize
        img_patch = img_patch.astype('float32')
        #gt_patch = gt_patch.astype('float32')
        for ch in range(len(self.min_channels)):
            img_patch[:,ch,...] = np.clip(img_patch[:,ch,...],self.min_channels[ch],self.max_channels[ch])
            img_patch[:,ch,...] = (img_patch[:,ch,...]-self.min_channels[ch]) / (self.max_channels[ch]-self.min_channels[ch])
        
        # if validation patch is not of patch_size (typically at border), add padding
        img_patch = np.pad(img_patch,
                           ((0,0),(0,0),
                            (left_padding,right_padding),
                            (top_padding,bottom_padding)),
                           "reflect")
        #gt_patch = np.pad(gt_patch,
                          #((0,0),(0,0),
                           #(left_padding,right_padding),
                           #(top_padding,bottom_padding)), 
                          #"reflect")
        img_patch = img_patch[:,:,:self.patch_size,:self.patch_size]
        #gt_patch = gt_patch[:,:,:self.patch_size,:self.patch_size]
        return img_patch #, gt_patch
        
    
    def choose_patches_for_iteration(self, batch_size):
#        annotation_tile = self.gt_train
#        num_gt_channels = len(self.class_id_set)

        if self.d_valid_pos_all_tiles is None and self.single_scheduler_mode:
            raise Exception("Can only call this method in single_scheduler_mode for rank=0")

        classes_have_data = []
        for class_id in self.d_valid_pos_all_tiles:
            if class_id > 0 and len(self.d_valid_pos_all_tiles[class_id][0]) > 0:
                classes_have_data.append(class_id)

        batch_patch_info = np.zeros((6, batch_size), np.int32)
        for batch in range(batch_size):
            class_id = classes_have_data[np.random.randint(0, len(classes_have_data))]
            positive_positions = self.d_valid_pos_all_tiles[class_id]
            idx_pos_rand = np.random.randint(0, positive_positions[0].shape[0])
            #print("class_id\t{}\tNbLabeledpixels\t{}".format(class_id, positive_positions[0].shape[0]))
            #print(positive_positions)
            # TODO : find the rectangle for the corresponding tile
            #idx_tile_id, tile_rectangle = self.__get_tile_id_for_position(
                #positive_positions[1][idx_pos_rand],
                #positive_positions[0][idx_pos_rand])

            #batch_patch_info[0, batch] = idx_tile_id # index of subtile on which the patch can be found
            batch_patch_info[0, batch] = positive_positions[2][idx_pos_rand] # rank on which the patch can be found
            batch_patch_info[1, batch] = positive_positions[0][idx_pos_rand] #left
            batch_patch_info[2, batch] = positive_positions[1][idx_pos_rand] #top
            batch_patch_info[3, batch] = class_id
            #batch_patch_info[4, batch] = positive_positions[0][idx_pos_rand] - tile_rectangle[0]
            #batch_patch_info[5, batch] = positive_positions[1][idx_pos_rand] - tile_rectangle[1]
            batch_patch_info[4, batch] = positive_positions[0][idx_pos_rand]
            batch_patch_info[5, batch] = positive_positions[1][idx_pos_rand]
        #print("batch_patch_info")
        #print(batch_patch_info)
        return batch_patch_info

    def generate_train_patch_fast(self, batch_size, batch_patch_info=None):
        if len(self.class_id_set) == 0:
            raise Exception("Set the desired classes first with set_multi_class")

        annotation_tile = self.gt_train
#        if self.dense_annotations is not None:
#            if np.random.randint(0, 5) > 0:
#                annotation_tile = self.gt_train
#            else:
#                annotation_tile = self.dense_annotations

        num_gt_channels = len(self.class_id_set)
        if self.add_contours_channel:
            num_gt_channels += 1

        if batch_patch_info is None:
            batch_patch_info = self.choose_patches_for_iteration(batch_size)

        num_in_batch = batch_patch_info.shape[1]

        all_img_patch = np.zeros((num_in_batch, self.tile.shape[0], self.patch_size, self.patch_size),  self.tile.dtype)

        all_gt_patch = np.zeros((num_in_batch,
                                 num_gt_channels,
                                 self.patch_size,
                                 self.patch_size), np.int8)


#        classes_have_data = []
#        for class_id in self.gt_train_valid_pos:
#            if class_id > 0 and len(self.gt_train_valid_pos[class_id][0]) > 0:
#                classes_have_data.append(class_id)

        for batch in range(num_in_batch):
#            class_id = classes_have_data[np.random.randint(0, len(classes_have_data))]
#            positive_positions = self.gt_train_valid_pos[class_id]

            #idx_pos_rand = np.random.randint(0, positive_positions[0].shape[0])
#           positive_positions[0][idx_pos_rand]
#positive_positions[1][idx_pos_rand]

            #get y and x that were generated and apply random translation
            dx = np.random.randint(-self.patch_size//3, self.patch_size//3)
            dy = np.random.randint(-self.patch_size//3, self.patch_size//3)

            r_top = batch_patch_info[4, batch] + dy - self.patch_size//2
            r_left = batch_patch_info[5, batch] + dx - self.patch_size//2
            class_id = batch_patch_info[3, batch]

#            print(r_tmp, r_left)
            r_top = np.clip(r_top, 0, annotation_tile.shape[0] - self.patch_size-1)
            r_left = np.clip(r_left, 0, annotation_tile.shape[1] - self.patch_size-1)

            img_patch = self.tile[:, r_top:(r_top+self.patch_size), r_left:(r_left + self.patch_size)]
#            print(r_top, r_left, self.gt_train.shape)
#            print( self.tile.shape, img_patch.shape)

            if img_patch.shape[1] != self.patch_size or img_patch.shape[2] != self.patch_size:
                raise Exception("CODING ERROR: patch has size h={} x w={}, sampled at y={},x={}, gt_size is h={}xw={}".format(
                    img_patch.shape[1],
                    img_patch.shape[2],
                    r_top,
                    r_left,
                    annotation_tile.shape[0],
                    annotation_tile.shape[1]))

            gt_patch = np.zeros((num_gt_channels, img_patch.shape[1], img_patch.shape[2]), np.int8)

            i_in_set = -1
            for i in range(len(self.class_id_set)):
                if self.class_id_set[i] == class_id:
                    i_in_set = i
                gt_patch_single_class = annotation_tile[r_top:(r_top+self.patch_size),
                                                        r_left:(r_left + self.patch_size)] == self.class_id_set[i]
                if self.compute_running_stats:
                    self.running_stats[self.class_id_set[i]] += np.sum(gt_patch_single_class)

                gt_patch[i,...] = gt_patch_single_class

                if self.add_contours_channel:
                    gt_patch[-1,...] += gt_patch[i,...]

            self.running_stats[0] += self.patch_size * self.patch_size - np.sum(gt_patch[-1,...])

            if self.add_contours_channel:
                gt_patch[-1,...] = 1 - gt_patch[-1,...]

            if i_in_set < 0:
                raise Exception("Class id {} is not part of the class id set {}!".format(class_id, self.class_id_set))

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
        
            # add image and gt to batch
            all_img_patch[batch,...] = img_patch
            all_gt_patch[batch,...] = gt_patch

        # convert to float and normalize
        all_img_patch = all_img_patch.astype('float32')
        all_gt_patch = all_gt_patch.astype('float32')
        for ch in range(len(self.min_channels)):
            # clip outlier values
            all_img_patch[:,ch,...] = np.clip(all_img_patch[:,ch,...],self.min_channels[ch],self.max_channels[ch])
            all_img_patch[:,ch,...] = (all_img_patch[:,ch,...]-self.min_channels[ch]) / (self.max_channels[ch]-self.min_channels[ch])

        if self.add_contours_channel:
            all_gt_patch[:,-1,...] = np.mean(all_img_patch, axis=1)
        # all_gt_patch[:,-1,...] = all_img_patch[:,0,...]
        return all_img_patch, all_gt_patch

    def get_running_stats(self):
        return self.running_stats

    def get_display_patch(self,y_start,y_stop,x_start,x_stop,channels):
        '''
        get a patch of arbitraty size and arbitrary bands of the ground truth
        '''
        gt_patch = self.gt_train[y_start:y_stop,x_start:x_stop]
        img_patch = self.tile[channels,y_start:y_stop,x_start:x_stop]
        img_patch = np.transpose(img_patch,(1,2,0))
        return gt_patch, img_patch
    
    def get_min_max_channels(self):
        return self.min_channels, self.max_channels
    
    def set_min_max_channels(self,min_channels,max_channels):
        self.min_channels = min_channels
        self.max_channels = max_channels

    def get_tile_sample_band(self, patch, sample_id, band_id):
        return None

    def get_class_ids(self):
        return self.class_id_set
