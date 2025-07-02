import itertools
import logging
import os
import os.path as osp
import tempfile
import re
import time
import glob
import json
import csv

import torch
import mmcv
import numpy as np
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
import torch.multiprocessing as mp
from tqdm import tqdm
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from .builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmcv.ops.nms import nms_rotated

from mmrotate.core import poly2obb_np
from .sodaa_eval.sodaa_eval import SODAAeval
import cv2


@DATASETS.register_module()
class SODAADataset(CustomDataset):
    CLASSES = ('airplane', 'helicopter', 'small-vehicle', 'large-vehicle',
               'ship', 'container', 'storage-tank', 'swimming-pool',
               'windmill')  # foreground categories only
    def __init__(self,
                 version,
                 ori_ann_file,
                 **kwargs):
        self.version = version
        super(SODAADataset, self).__init__(**kwargs)
        self.ori_data_infos = self.load_ori_annotations(ori_ann_file)
        self.cat_ids = self._get_cat_ids()


    def __len__(self):
        """Total number of samples of data."""
        return len(self.data_infos)

    def _get_cat_ids(self):
        cat_ids = dict()
        for idx, cat in enumerate(self.CLASSES):
            cat_ids[idx] = cat
        return cat_ids


    def load_ori_annotations(self, ori_ann_folder):
        """ Load annotation info of raw images. """
        ann_files = glob.glob(ori_ann_folder + '/*.json')
        ori_data_infos = []
        for ann_file in ann_files:
            data_info = {}
            img_name = ann_file.replace('.json', '.jpg').split(os.sep)[-1]
            data_info['filename'] = img_name
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            if os.path.getsize(ann_file) == 0:
                continue

            f = json.load(open(ann_file, 'r'))
            annotations = f['annotations']

            for ann in annotations:
                poly = np.array(ann['poly'], dtype=np.float32)
                if len(poly) > 8:
                    continue    # neglect those objects annotated with more than 8 polygons
                try:
                    x, y, w, h, a = poly2obb_np(poly, self.version)
                except:  # noqa: E722
                    continue
                label = int(ann['category_id'])  # 0-index
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                      dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)

            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)

            ori_data_infos.append(data_info)
        self.ori_img_ids = [*map(lambda x: x['filename'].split(os.sep)[-1][:-4], ori_data_infos)]
        return ori_data_infos

    def get_ori_ann_info(self, idx):
        """Get annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        return self.ori_data_infos[idx]['ann']

    def load_annotations(self, ann_folder):
        """Load annotation from COCO style annotation file.

        Args:
            ann_folder (str): Directory that contains annotation file of SODA-A dataset.

        """
        cls_map = {c: i for i, c in enumerate(self.CLASSES)}  # 0-index
        ann_files = glob.glob(ann_folder + '/*.json')
        data_infos = []
        for ann_file in ann_files:
            data_info = {}
            img_name = ann_file.replace('.json', '.jpg').split(os.sep)[-1]
            data_info['filename'] = img_name
            data_info['ann'] = {}
            gt_bboxes = []
            gt_labels = []
            gt_polygons = []
            gt_bboxes_ignore = []
            gt_labels_ignore = []
            gt_polygons_ignore = []

            if os.path.getsize(ann_file) == 0:
                continue

            f = json.load(open(ann_file, 'r'))
            annotations = f['annotations']

            for ann in annotations:
                poly = np.array(ann['poly'], dtype=np.float32)
                try:
                    x, y, w, h, a = poly2obb_np(poly, self.version)
                except:  # noqa: E722
                    continue
                label = int(ann['cat_id'])  # 0-index
                trunc = int(ann['trunc'])
                gt_bboxes.append([x, y, w, h, a])
                gt_labels.append(label)
                gt_polygons.append(poly)

            if gt_bboxes:
                data_info['ann']['bboxes'] = np.array(
                    gt_bboxes, dtype=np.float32)
                data_info['ann']['labels'] = np.array(
                    gt_labels, dtype=np.int64)
                data_info['ann']['polygons'] = np.array(
                    gt_polygons, dtype=np.float32)
            else:
                data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                      dtype=np.float32)
                data_info['ann']['labels'] = np.array([], dtype=np.int64)
                data_info['ann']['polygons'] = np.zeros((0, 8),
                                                        dtype=np.float32)

            if gt_polygons_ignore:
                data_info['ann']['bboxes_ignore'] = np.array(
                    gt_bboxes_ignore, dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    gt_labels_ignore, dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.array(
                    gt_polygons_ignore, dtype=np.float32)
            else:
                data_info['ann']['bboxes_ignore'] = np.zeros(
                    (0, 5), dtype=np.float32)
                data_info['ann']['labels_ignore'] = np.array(
                    [], dtype=np.int64)
                data_info['ann']['polygons_ignore'] = np.zeros(
                    (0, 8), dtype=np.float32)

            data_infos.append(data_info)
        self.img_ids = [*map(lambda x: x['filename'].split(os.sep)[-1][:-4], data_infos)]
        return data_infos


    def _filter_imgs(self):
        """Filter images without ground truths."""
        valid_inds = []
        for i, data_info in enumerate(self.data_infos):
            if data_info['ann']['labels'].size > 0:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        All set to 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)


    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.sodaa.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.sodaa.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def evaluate(self,
                 results,
                 metric='mAP',
                 logger=None,
                 proposal_nums=(100, 300, 1000),
                 iou_thr=None,
                 scale_ranges=None,
                 metric_items=None,
                 nproc=4):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thr (float | list[float]): IoU threshold. It must be a float
                when evaluating mAP, and can be a list when evaluating recall.
                Default: 0.5.
            scale_ranges (list[tuple] | None): Scale ranges for evaluating mAP.
                Default: None.
            nproc (int): Processes used for computing TP and FP.
                Default: 4.
        """
        nproc = 4
        merged_results = self.merge_det(results, nproc=nproc)
        merge_idx = [self.ori_img_ids.index(res[0]) for res in merged_results]
        results = [res[1] for res in merged_results]    # exclude `id` for evaluation

        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f'metric {metric} is not supported')
        annotations = [self.get_ori_ann_info(i) for i in merge_idx]
        # evaluation
        if iou_thr is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)

        eval_results = {}
        SODAAEval = SODAAeval(annotations, results, numCats=len(self.CLASSES), nproc=nproc)
        SODAAEval.params.iouThrs = iou_thrs

        # mapping of cocoEval.stats
        sodaa_metric_names = {
            'AP': 0,
            'AP_50': 1,
            'AP_75': 2,
            'AP_eS': 3,
            'AP_rS': 4,
            'AP_gS': 5,
            'AP_Normal': 6,
            'AR@20000': 7,
            'AR_eS@20000': 8,
            'AR_rS@20000': 9,
            'AR_gS@20000': 10,
            'AR_Normal@20000': 11
        }
        SODAAEval.evaluate()
        SODAAEval.accumulate()
        SODAAEval.summarize()

        classwise = True
        if classwise:  # Compute per-category AP
            # Compute per-category AP
            # from https://github.com/facebookresearch/detectron2/
            precisions = SODAAEval.eval['precision']
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for catId, catName in self.cat_ids.items():
                # area range index 0: all area ranges
                # max dets index -1: typically 20000 per image
                precision = precisions[:, :, catId, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{catName}', f'{float(ap):0.3f}'))

            num_columns = min(6, len(results_per_category) * 2)
            results_flatten = list(
                itertools.chain(*results_per_category))
            headers = ['category', 'AP'] * (num_columns // 2)
            results_2d = itertools.zip_longest(*[
                results_flatten[i::num_columns]
                for i in range(num_columns)
            ])
            table_data = [headers]
            table_data += [result for result in results_2d]
            table = AsciiTable(table_data)
            txtFile = open(txtPth, 'a+')
            txtFile.writelines(f"\n{'-' * 30}Class-wise Evaluation{'-' * 30}\n")
            txtFile.writelines(table.table)
            txtFile.writelines("\n")
            txtFile.close()
            print_log('\n' + table.table, logger=logger)

        # TODO: proposal evaluation
        if metric_items is None:
            metric_items = [
                'AP', 'AP_50', 'AP_75', 'AP_eS',
                'AP_rS', 'AP_gS', 'AP_Normal'
            ]

        for metric_item in metric_items:
            key = f'{metric}_{metric_item}'
            val = float(
                f'{SODAAEval.stats[sodaa_metric_names[metric_item]]:.3f}'
            )
            eval_results[key] = val
        ap = SODAAEval.stats[:7]
        eval_results[f'{metric}_mAP_copypaste'] = (
            f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} '
            f'{ap[3]:.3f} {ap[4]:.3f} {ap[5]:.3f} '
            f'{ap[6]:.3f} '
        )
        return eval_results

    def translate(self, bboxes, x, y):
        translated = bboxes.copy()
        translated[..., :2] = translated[..., :2] + \
                              np.array([x, y], dtype=np.float32)
        return translated

    def merge_det(self,
                  results,
                  with_merge=True,
                  nms_iou_thr=0.5,
                  nproc=10,
                  save_dir=None,
                  **kwargs):
        if mmcv.is_list_of(results, tuple):
            dets, segms = results
        else:
            dets = results

        if not with_merge:
            results = [(data_info['id'], result)
                       for data_info, result in zip(self.data_infos, results)]
            if save_dir is not None:
                pass  # TODO:
            return results

        if mp.get_start_method(allow_none=True) != 'spawn':
            print("INFO: Multiprocessing start method is not 'spawn'. Attempting to set it...")
            try:
                mp.set_start_method('spawn', force=True)
                print("SUCCESS: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                print(f"WARNING: Could not set start method to 'spawn': {e}")
                print(
                    "WARNING: This may lead to CUDA errors. Please consider setting it at the start of your main script.")

        print('\n>>> Merge detected results of patch for whole image evaluating...')
        start_time = time.time()

        # Use a list to store intermediate results, then concatenate
        collector = defaultdict(list)

        # Pre-process detections for each patch
        for data_info, result in tqdm(zip(self.data_infos, dets), total=len(self.data_infos),
                                      desc="Collecting patch results"):
            filename = data_info['filename']
            # Improved file name parsing: Use a more robust split, or even better,
            # store coordinates in `data_info` during data loading.
            try:
                parts = filename.split('___')
                x_start, y_start = \
                    int(parts[0].split('__')[-1]), \
                    int(parts[-1].split('.')[0])
                ori_name = filename.split('__')[0]
            except IndexError:
                print(f"Warning: Could not parse coordinates from filename: {filename}. Skipping.")
                continue

            # This part is optimized for better array concatenation
            all_bboxes_for_patch = []
            for i, res in enumerate(result):
                if res.size > 0:
                    bboxes, scores = res[:, :-1], res[:, [-1]]
                    bboxes = self.translate(bboxes, x_start, y_start)
                    labels = np.full((bboxes.shape[0], 1), i, dtype=np.float32)
                    all_bboxes_for_patch.append(np.concatenate(
                        [labels, bboxes, scores], axis=1
                    ))

            if all_bboxes_for_patch:
                new_result = np.concatenate(all_bboxes_for_patch, axis=0)
                collector[ori_name].append(new_result)

        # NMS device is default to 'cuda:0' otherwise 'cpu'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for NMS: {device}")

        # Use functools.partial for better function wrapping
        merge_func = partial(merge_function, CLASSES=self.CLASSES, iou_thr=nms_iou_thr, device=device)

        # Process results in parallel or sequentially
        if nproc > 1:
            try:
                with Pool(nproc) as pool:
                    merged_results = pool.map(merge_func, list(collector.items()))
            except Exception as e:
                print(f"Error during multiprocessing: {e}. Falling back to sequential processing.")
                merged_results = list(map(merge_func, list(collector.items())))
        else:
            merged_results = list(map(merge_func, list(collector.items())))

        if save_dir is not None:
            pass

        stop_time = time.time()
        print('Merge results completed, it costs %.1f seconds.' % (stop_time - start_time))
        return merged_results

def merge_function(info, CLASSES, iou_thr, device='cpu'):
    """
    Merging function for a single image  with GPU support
    """
    img_id, label_dets = info
    # Concatenate all detection results for the current image in one go
    label_dets = np.concatenate(label_dets, axis=0)

    if len(label_dets) == 0:
        return img_id, [np.empty((0, 5), dtype=np.float32) for _ in CLASSES]

    # Extract labels and detections
    labels = label_dets[:, 0]
    dets = label_dets[:, 1:]

    ori_img_results = []

    # Convert numpy arrays to torch tensors once, and move to device
    all_bboxes = torch.from_numpy(dets[:, :-1]).to(device).contiguous()
    all_scores = torch.from_numpy(dets[:, -1]).to(device).contiguous()
    all_labels = torch.from_numpy(labels).to(device).to(torch.int64)

    # Sanity check for bounding box format
    if all_bboxes.shape[1] == 4:
        print("Warning: Bounding box format is (N, 4). `nms_rotated` expects (N, 5).")
        pass
    elif all_bboxes.shape[1] == 5:
        pass
    else:
        raise ValueError(f"Unsupported bounding box shape: {all_bboxes.shape[1]}")

    for i in range(len(CLASSES)):
        # Filter detections for the current class efficiently using boolean indexing
        cls_mask = (all_labels == i)

        # If no detections for this class, add an empty array
        if not torch.any(cls_mask):
            ori_img_results.append(np.empty((0, dets.shape[1]), dtype=np.float32))
            continue

        cls_bboxes = all_bboxes[cls_mask]
        cls_scores = all_scores[cls_mask]

        # Apply NMS on the selected detections using the MMCV CUDA operator
        # This is where the major speed-up comes from.
        results, inds = nms_rotated(cls_bboxes, cls_scores, iou_thr)

        # Move results back to CPU and convert to numpy for returning
        results_np = results.cpu().numpy()
        ori_img_results.append(results_np)

    return img_id, ori_img_results
