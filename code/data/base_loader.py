import logging
import torch
import numpy as np
import MinkowskiEngine as ME


class CollationFunctionFactory:
    def __init__(self, collation_type='default'):
        if collation_type == 'default':
            self.collation_fn = self.collate_default
        elif collation_type == 'collate_pair':
            self.collation_fn = self.collate_pair_fn
        else:
            raise ValueError(f'collation_type {collation_type} not found')

    def __call__(self, list_data):
        return self.collation_fn(list_data)

    def collate_default(self, list_data):
        return list_data

    def collate_pair_fn(self, list_data):
        N = len(list_data)
        list_data = [data for data in list_data if data is not None]
        if N != len(list_data):
            logging.info(f"Retain {len(list_data)} from {N} data.")

        if len(list_data) == 0:
            raise ValueError('No data in the batch')

        coords, feats, coords_s8, feats_s8, rot, pose = list(zip(*list_data))

        coords_batch = ME.utils.batched_coordinates(coords)
        coords_s8_batch = ME.utils.batched_coordinates(coords_s8)
        feats_batch = torch.from_numpy(np.concatenate(feats, 0)).float()
        feats_s8_batch = torch.from_numpy(np.concatenate(feats_s8, 0)).float()
        rot_batch = torch.from_numpy(np.stack(rot)).float()
        pose_batch = torch.from_numpy(np.stack(pose)).float()

        return {
            'sinput_C': coords_batch,
            'sinput_F': feats_batch,
            'sinput_s8_C': coords_s8_batch,
            'sinput_s8_F': feats_s8_batch,
            'rot': rot_batch,
            'pose': pose_batch
        }
