import logging
import os
from os import listdir
from os.path import splitext, isfile, join
from .data_loading import BasicDataset

class Volume1Dataset(BasicDataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '_mask'):
        # Filter for volume 1 files before initialization
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        
        # Get only volume 1 files (they all start with 'volume_1_' and end with '.png')
        self.ids = [splitext(file)[0] for file in listdir(images_dir) 
                   if isfile(join(images_dir, file)) 
                   and not file.startswith('.')
                   and file.startswith('volume_1_')
                   and file.endswith('.png')]  # Volume 1 PNG files only
        
        if not self.ids:
            raise RuntimeError(f'No volume 1 images found in {images_dir}')
            
        logging.info(f'Found {len(self.ids)} volume 1 images')
        
        # Initialize the rest of the dataset
        super().__init__(images_dir, mask_dir, scale, mask_suffix) 