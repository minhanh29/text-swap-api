import os
from time import time
import sys

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import torch
import tqdm
import yaml
from omegaconf import OmegaConf
from torch.utils.data._utils.collate import default_collate
import pytorch_lightning as ptl

from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import MyDataset
from saicinpainting.training.modules.ffc import FFCResNetGenerator


class InpaintingModel(ptl.LightningModule):
    def __init__(self, config, *args, concat_mask=True, rescale_scheduler_kwargs=None, image_to_discriminator='predicted_image',
                 add_noise_kwargs=None, noise_fill_hole=False, const_area_crop_kwargs=None,
                 distance_weighter_kwargs=None, distance_weighted_mask_for_discr=False,
                 fake_fakes_proba=0, fake_fakes_generator_kwargs=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.concat_mask = concat_mask
        self.generator = FFCResNetGenerator(**config.generator)

    def forward(self, img, mask):
        masked_img = img * (1 - mask)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        predicted_image = self.generator(masked_img)
        result = mask * predicted_image + (1 - mask) * img

        return result


class Inpainter:
    def __init__(self, model_path):
        self.device = torch.device("cpu")

        train_config_path = os.path.join(model_path, 'config.yml')
        with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))

        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'

        checkpoint_path = os.path.join(model_path, "lama.pth")
        # checkpoint_path = os.path.join(model_path, "best.ckpt")
        self.model = InpaintingModel(train_config)
        state = torch.load(checkpoint_path, map_location="cpu")
        self.model.load_state_dict(state['state_dict'], strict=False)
        self.model.freeze()
        self.model.to(self.device)

    def predict(self, img_list, mask_list):
        dataset = MyDataset(img_list, mask_list, pad_out_to_modulo=8)
        result = []
        with torch.no_grad():
            for img_i in range(len(dataset)):
                batch = move_to_device(default_collate([dataset[img_i]]), self.device)
                batch['mask'] = (batch['mask'] > 0) * 1
                pred = self.model(batch["image"], batch["mask"])
                cur_res = pred[0].permute(1, 2, 0).detach().cpu().numpy()

                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

                # RGB image
                cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
                result.append(cur_res)
        return result
