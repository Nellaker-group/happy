from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.utils.data import IterableDataset
from PIL import Image
import torchvision.transforms as transforms

from happy.utils.utils import process_image

import sys
import pickle 
import time

class MSDataset(IterableDataset, ABC):
    def __init__(self, microscopefile, remaining_data, big_tile_size,transform=None):
        self.file = microscopefile
        self.remaining_data = remaining_data
        self.tile_size= big_tile_size
        self.transform = transform

        self.target_width = self.file.target_tile_width
        self.target_height = self.file.target_tile_height
        self.rescale_ratio = self.file.rescale_ratio
        self.mask_array = self.file.mask_array
        self.start = 0
        self.end = len(self.remaining_data)

    # called by a dataloader. Uses torch workers to load data onto a gpu
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return self._iter_data(self.start, self.end)
        else:
            # splits the datasets each worker gets proportional to the number of workers
            per_worker = int(
                np.math.ceil((self.end - self.start) / float(worker_info.num_workers))
            )
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)

            return self._iter_data(iter_start, iter_end)

    @abstractmethod
    def _iter_data(self, iter_start, iter_end):
        pass

    @abstractmethod
    def _get_dataset_section(self, target_w, target_h, tile_range):
        pass


class NucleiDataset(MSDataset):
    def _iter_data(self, iter_start, iter_end):
        # print(f'rescale_ratio = {self.rescale_ratio}')
        # get small nuc tile xys from nuclei referencing results
        nuc_tile_coord = self.remaining_data[iter_start: iter_end] 
        # get big nuc tile xys based on the small tiles xys
        big_nuc_tile_xys = self.file.get_nuc_big_tile_xys(nuc_tile_coord, self.tile_size,self.target_width,self.target_height) 

        #for each big tile:
        for big_nuc_tile_coord in big_nuc_tile_xys:
            #big_load_st = time.time()
        
            big_img = self.file.get_big_tile_by_coords_nuc(
                big_nuc_tile_coord[0], big_nuc_tile_coord[1], self.tile_size
            ) 

            #big_load_end = time.time()
            #print(f"big tile loading took: {big_load_end - big_load_st:.3f} seconds")

            # select the small tiles within that big tiles
            coord_df= self.file.small_tile_xy_big_tile(nuc_tile_coord,big_nuc_tile_coord,self.tile_size,
                                                       self.target_width, self.target_height)

            #get the cropped small tile from big tile
            for img, tile_index, empty_tile in self._get_dataset_section(
                big_xy=big_nuc_tile_coord,
                target_w=self.target_width,
                target_h=self.target_height,
                coord_df=coord_df,
                big_tile_tensor=big_img
            ):
                #yield_nuc_st = time.time()
                if not empty_tile:
                    img = process_image(img).astype(np.float32) / 255.0     
                    
                sample = {
                    "img": img,
                    "tile_index": tile_index,
                    "empty_tile": empty_tile,
                    "scale": None,
                    "annot": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                }
                if self.transform and not empty_tile:
                    sample = self.transform(sample)
                yield sample
                #yield_nuc_end = time.time()
                #print(f"yeild small nucs in one big tile took: {yield_nuc_end - yield_nuc_st:.3f} seconds")

    # Generator to create a datasets for small nuc tiles within the big tiles 
    def _get_dataset_section(self, big_xy, target_w, target_h,coord_df, big_tile_tensor):

        tile_coords = coord_df #small tile coordinate df, columns: [index, x, y]
        # print(f'number of tiles in big tile = {len(tile_coords)}')
        full_image_tensor = big_tile_tensor #big tile tensor
        big_tile_x=big_xy[0]
        big_tile_y=big_xy[1]

        for row in tile_coords.itertuples(index=False):
            
            wsi_x= row.x
            wsi_y= row.y
            #transferring small tile xy from WSI reference frame to big tile reference frame
            x = self.file.wsi_to_tile_nuclei(big_tile_x, wsi_x) 
            y = self.file.wsi_to_tile_nuclei(big_tile_y, wsi_y)
            wsi_xy = (wsi_x,wsi_y)
            #check if small tile is empty, return small tile + index + empty status
            if self.file.check_nuc_tile_blank(wsi_xy,self.mask_array):
                #print('empty')
                yield None, row.index, True
            else:
                #crop small tile from big tile
                #crop_st=time.time()
                img = full_image_tensor[int(y):int(y+target_h),
                                        int(x):int(x+target_w),:]
                assert(img.shape ==(1200,1600,3)),f"Found image with shape {img.shape}"
                yield img, row.index, False
                #crop_end=time.time()
                #print(f"small tile cropping took: {crop_end - crop_st:.3f} seconds")


class CellDataset(MSDataset):
    def _iter_data(self, iter_start, iter_end):
        cell_coord = self.remaining_data[iter_start: iter_end] # get cell xys from nuclei referencing results
        big_tile_xys = self.file.get_big_tile_xy(cell_coord,tile_tilesize=self.tile_size,tile_overlap=224) # get all top left xys for big tiles
        
        # iterate through each big tile
        for big_tile_coord in big_tile_xys:

            # load big tile to memory (already tensor form)
            big_img = self.file.get_big_tile_by_coords_cell(
                big_tile_coord[0], big_tile_coord[1], self.tile_size
            ) 

            # get centriod xys of each cell within the big tile range (related to WSI coordinate)
            coord_list = self.file.get_xys_in_big_tile(big_tile_coord,cell_coord,tile_tilesize=self.tile_size,tile_overlap=224)

            # iterate through each cell image within the big tile
            for img, coord in self._get_dataset_section(big_xy=big_tile_coord,
                                                        cell_tile_size=224, 
                                                        coord_list=coord_list,
                                                        big_tile_tensor=big_img):
                
                if self.transform:
                    sample = self.transform(image=img)
                    sample["img"] = sample["image"]
                    del sample["image"]
                    sample["coord"] = np.array(coord)
                    sample["annot"] = np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])
                else:
                    sample = {
                        "img": img,
                        "coord": np.array(coord),
                        "annot": np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
                    }

                yield sample

    def _get_dataset_section(self, big_xy, cell_tile_size, coord_list, big_tile_tensor):
        cell_coords_wsi = coord_list
        full_image_tensor = big_tile_tensor
        big_tile_x=big_xy[0]
        big_tile_y=big_xy[1]
        
        # iterate through each cell xy within one big tile
        for _dict in cell_coords_wsi:
            
            # convert WSI xys to Big tile related xys (Big tile top left xys as (0,0) + resizing for ResNet)
            wsi_x= _dict[0]
            wsi_y= _dict[1]
            x=self.file.wsi_to_tile_cell(big_tile_x,wsi_x)
            y=self.file.wsi_to_tile_cell(big_tile_y,wsi_y)

            # crop cell images tensor out from the big tile
            img = full_image_tensor[int(y-cell_tile_size/2):int(y+cell_tile_size/2),
                                    int(x-cell_tile_size/2):int(x+cell_tile_size/2),:]

            # cell img tensor shape chack
            assert img.shape == (224, 224, 3), f"Found image with shape {img.shape}"
            
            yield img, (wsi_x, wsi_y)
