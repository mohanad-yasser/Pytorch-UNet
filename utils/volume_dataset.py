import logging
import os
from os import listdir
from os.path import splitext, isfile, join
from .data_loading import BasicDataset

class VolumeDataset(BasicDataset):
    def __init__(self, images_dir: str, mask_dir: str, volume_name: str, scale: float = 1.0, mask_suffix: str = '_mask', transform=None):
        # Filter for specific volume files before initialization
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.volume_name = volume_name
        
        # Get only files for the specified volume
        self.ids = [splitext(file)[0] for file in listdir(images_dir) 
                   if isfile(join(images_dir, file)) 
                   and not file.startswith('.')
                   and file.startswith(volume_name)
                   and file.endswith('.png')]
        
        if not self.ids:
            raise RuntimeError(f'No {volume_name} images found in {images_dir}')
            
        logging.info(f'Found {len(self.ids)} {volume_name} images')
        
        # Initialize the rest of the dataset
        super().__init__(images_dir, mask_dir, scale, mask_suffix, transform=transform) 