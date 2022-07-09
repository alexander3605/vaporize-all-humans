import logging
import math
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import tqdm
from torch.utils.data import DataLoader

from external.lama.saicinpainting.evaluation.utils import move_to_device

LOGGER = logging.getLogger(__name__)


class InpaintingEvaluator():
    def __init__(self, dataset, scores, area_grouping=True, bins=10, batch_size=32, device='cuda',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param dataset: torch.utils.data.Dataset which contains images and masks
        :param scores: dict {score_name: EvaluatorScore object}
        :param area_grouping: in addition to the overall scores, allows to compute score for the groups of samples
            which are defined by share of area occluded by mask
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param batch_size: batch_size for the dataloader
        :param device: device to use
        """
        self.scores = scores
        self.dataset = dataset

        self.area_grouping = area_grouping
        self.bins = bins

        self.device = torch.device(device)

        self.dataloader = DataLoader(self.dataset, shuffle=False, batch_size=batch_size)

        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range

    def _get_bin_edges(self):
        bin_edges = np.linspace(0, 1, self.bins + 1)

        num_digits = max(0, math.ceil(math.log10(self.bins)) - 1)
        interval_names = []
        for idx_bin in range(self.bins):
            start_percent, end_percent = round(100 * bin_edges[idx_bin], num_digits), \
                                         round(100 * bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            interval_names.append("{0}-{1}%".format(start_percent, end_percent))

        groups = []
        for batch in self.dataloader:
            mask = batch['mask']
            batch_size = mask.shape[0]
            area = mask.to(self.device).reshape(batch_size, -1).mean(dim=-1)
            bin_indices = np.searchsorted(bin_edges, area.detach().cpu().numpy(), side='right') - 1
            # corner case: when area is equal to 1, bin_indices should return bins - 1, not bins for that element
            bin_indices[bin_indices == self.bins] = self.bins - 1
            groups.append(bin_indices)
        groups = np.hstack(groups)

        return groups, interval_names

    def evaluate(self, model=None):
        """
        :param model: callable with signature (image_batch, mask_batch); should return inpainted_batch
        :return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        results = dict()
        if self.area_grouping:
            groups, interval_names = self._get_bin_edges()
        else:
            groups = None

        for score_name, score in tqdm.auto.tqdm(self.scores.items(), desc='scores'):
            score.to(self.device)
            with torch.no_grad():
                score.reset()
                for batch in tqdm.auto.tqdm(self.dataloader, desc=score_name, leave=False):
                    batch = move_to_device(batch, self.device)
                    image_batch, mask_batch = batch['image'], batch['mask']
                    if self.clamp_image_range is not None:
                        image_batch = torch.clamp(image_batch,
                                                  min=self.clamp_image_range[0],
                                                  max=self.clamp_image_range[1])
                    if model is None:
                        assert 'inpainted' in batch, \
                            'Model is None, so we expected precomputed inpainting results at key "inpainted"'
                        inpainted_batch = batch['inpainted']
                    else:
                        inpainted_batch = model(image_batch, mask_batch)
                    score(inpainted_batch, image_batch, mask_batch)
                total_results, group_results = score.get_value(groups=groups)

            results[(score_name, 'total')] = total_results
            if groups is not None:
                for group_index, group_values in group_results.items():
                    group_name = interval_names[group_index]
                    results[(score_name, group_name)] = group_values

        if self.integral_func is not None:
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

        return results


def ssim_fid100_f1(metrics, fid_scale=100):
    ssim = metrics[('ssim', 'total')]['mean']
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * ssim * fid_rel / (ssim + fid_rel + 1e-3)
    return f1


def lpips_fid100_f1(metrics, fid_scale=100):
    neg_lpips = 1 - metrics[('lpips', 'total')]['mean']  # invert, so bigger is better
    fid = metrics[('fid', 'total')]['mean']
    fid_rel = max(0, fid_scale - fid) / fid_scale
    f1 = 2 * neg_lpips * fid_rel / (neg_lpips + fid_rel + 1e-3)
    return f1



class InpaintingEvaluatorOnline(nn.Module):
    def __init__(self, scores, bins=10, image_key='image', inpainted_key='inpainted',
                 integral_func=None, integral_title=None, clamp_image_range=None):
        """
        :param scores: dict {score_name: EvaluatorScore object}
        :param bins: number of groups, partition is generated by np.linspace(0., 1., bins + 1)
        :param device: device to use
        """
        super().__init__()
        LOGGER.info(f'{type(self)} init called')
        self.scores = nn.ModuleDict(scores)
        self.image_key = image_key
        self.inpainted_key = inpainted_key
        self.bins_num = bins
        self.bin_edges = np.linspace(0, 1, self.bins_num + 1)

        num_digits = max(0, math.ceil(math.log10(self.bins_num)) - 1)
        self.interval_names = []
        for idx_bin in range(self.bins_num):
            start_percent, end_percent = round(100 * self.bin_edges[idx_bin], num_digits), \
                                         round(100 * self.bin_edges[idx_bin + 1], num_digits)
            start_percent = '{:.{n}f}'.format(start_percent, n=num_digits)
            end_percent = '{:.{n}f}'.format(end_percent, n=num_digits)
            self.interval_names.append("{0}-{1}%".format(start_percent, end_percent))

        self.groups = []

        self.integral_func = integral_func
        self.integral_title = integral_title
        self.clamp_image_range = clamp_image_range

        LOGGER.info(f'{type(self)} init done')

    def _get_bins(self, mask_batch):
        batch_size = mask_batch.shape[0]
        area = mask_batch.view(batch_size, -1).mean(dim=-1).detach().cpu().numpy()
        bin_indices = np.clip(np.searchsorted(self.bin_edges, area) - 1, 0, self.bins_num - 1)
        return bin_indices

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Calculate and accumulate metrics for batch. To finalize evaluation and obtain final metrics, call evaluation_end
        :param batch: batch dict with mandatory fields mask, image, inpainted (can be overriden by self.inpainted_key)
        """
        result = {}
        with torch.no_grad():
            image_batch, mask_batch, inpainted_batch = batch[self.image_key], batch['mask'], batch[self.inpainted_key]
            if self.clamp_image_range is not None:
                image_batch = torch.clamp(image_batch,
                                          min=self.clamp_image_range[0],
                                          max=self.clamp_image_range[1])
            self.groups.extend(self._get_bins(mask_batch))

            for score_name, score in self.scores.items():
                result[score_name] = score(inpainted_batch, image_batch, mask_batch)
        return result

    def process_batch(self, batch: Dict[str, torch.Tensor]):
        return self(batch)

    def evaluation_end(self, states=None):
        """:return: dict with (score_name, group_type) as keys, where group_type can be either 'overall' or
            name of the particular group arranged by area of mask (e.g. '10-20%')
            and score statistics for the group as values.
        """
        LOGGER.info(f'{type(self)}: evaluation_end called')

        self.groups = np.array(self.groups)

        results = {}
        for score_name, score in self.scores.items():
            LOGGER.info(f'Getting value of {score_name}')
            cur_states = [s[score_name] for s in states] if states is not None else None
            total_results, group_results = score.get_value(groups=self.groups, states=cur_states)
            LOGGER.info(f'Getting value of {score_name} done')
            results[(score_name, 'total')] = total_results

            for group_index, group_values in group_results.items():
                group_name = self.interval_names[group_index]
                results[(score_name, group_name)] = group_values

        if self.integral_func is not None:
            results[(self.integral_title, 'total')] = dict(mean=self.integral_func(results))

        LOGGER.info(f'{type(self)}: reset scores')
        self.groups = []
        for sc in self.scores.values():
            sc.reset()
        LOGGER.info(f'{type(self)}: reset scores done')

        LOGGER.info(f'{type(self)}: evaluation_end done')
        return results
