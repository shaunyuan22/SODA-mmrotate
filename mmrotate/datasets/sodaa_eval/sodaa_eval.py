#==============================================================================================
#
# Author: shaunyuan (shaunyuan@mail.nwpu.edu.cn)
#
# Description:
#     The official evaluation script for SODA-A dataset, which originates from COCO evaluator.
#     This tool enhances the official COCO evaluation script by introducing multi-processing
#     and multi-GPU parallelism. This results in a dramatic speedup for the evaluation process,
#     particularly when dealing with images containing a high density of instances.
#
#==============================================================================================
# Note:
#     This project depends on the third-library: mmcv
#==============================================================================================


__author__ = 'shaunyuan (shaunyuan@mail.nwpu.edu.cn)'

import copy
import datetime
import time
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Pool, Manager
import multiprocessing as mp
from functools import partial

import numpy as np
import torch
from mmcv.ops import box_iou_rotated


def init_worker_with_counter(counter, devices_list):
    global worker_device
    worker_id = counter.value
    counter.value += 1

    worker_device = devices_list[worker_id % len(devices_list)]


def top_level_compute_iou_worker(args):
    global worker_device
    device = worker_device

    imgId, catId, gts_data, dts_data, params_dict = args

    gt = gts_data.get((imgId, catId), [])
    dt = dts_data.get((imgId, catId), [])

    if len(gt) == 0 or len(dt) == 0:
        return (imgId, catId, np.empty((0, 0), dtype=np.float32))

    scores_tensor = torch.as_tensor([d['score'] for d in dt], device=device)

    inds = torch.argsort(scores_tensor, descending=True)

    max_dets = params_dict['maxDets'][-1]
    if len(inds) > max_dets:
        inds = inds[:max_dets]

    if inds.numel() == 0:
        return (imgId, catId, np.empty((0, 0), dtype=np.float32))

    inds_np = np.argsort([-d['score'] for d in dt], kind='mergesort')
    dt = [dt[i] for i in inds_np]
    if len(dt) > max_dets:
        dt = dt[0:max_dets]
        inds_np = inds_np[0:max_dets]

    g_boxes = torch.tensor([g['bbox'] for g in gt], dtype=torch.float32, device=device)
    d_boxes = torch.tensor([d['bbox'] for d in dt], dtype=torch.float32, device=device)

    ious_tensor = box_iou_rotated(d_boxes, g_boxes)
    ious_np = ious_tensor.cpu().numpy()

    full_ious_np = np.zeros((len(dts_data.get((imgId, catId), [])), len(gt)), dtype=np.float32)
    full_ious_np[inds_np, :] = ious_np

    return (imgId, catId, full_ious_np)


class SODAAeval:
    # Interface for evaluating detection on the SODA-A dataset.
    # This evaluation code originates from COCO dataset.
    #
    # The usage for CocoEval is as follows:
    #  sodaaGt=..., sodaaDt=...       # load dataset and results
    #  E = CocoEval(sodaaGt,sodaaDt); # initialize CocoEval object
    #  E.params.recThrs = ...;      # set parameters as desired
    #  E.evaluate();                # run per image evaluation
    #  E.accumulate();              # accumulate per image results
    #  E.summarize();               # display summary metrics of results
    # For example usage see evalDemo.m and http://mscoco.org/.
    #
    # The evaluation parameters are as follows (defaults in brackets):
    #  imgIds     - [all] N img ids to use for evaluation
    #  catIds     - [all] K cat ids to use for evaluation
    #  iouThrs    - [.5:.05:.95] T=10 IoU thresholds for evaluation
    #  recThrs    - [0:.01:1] R=101 recall thresholds for evaluation
    #  areaRng    - [...] A=4 object area ranges for evaluation
    #  maxDets    - [1 10 100] M=3 thresholds on max detections per image
    #  iouType    - ['segm'] set iouType to 'segm', 'bbox' or 'keypoints'
    #  iouType replaced the now DEPRECATED useSegm parameter.
    #  useCats    - [1] if true use category labels for evaluation
    # Note: if useCats=0 category labels are ignored as in proposal scoring.
    # Note: multiple areaRngs [Ax2] and maxDets [Mx1] can be specified.
    #
    # evaluate(): evaluates detections on every image and every category and
    # concats the results into the "evalImgs" with fields:
    #  dtIds      - [1xD] id for each of the D detections (dt)
    #  gtIds      - [1xG] id for each of the G ground truths (gt)
    #  dtMatches  - [TxD] matching gt id at each IoU or 0
    #  gtMatches  - [TxG] matching dt id at each IoU or 0
    #  dtScores   - [1xD] confidence of each dt
    #  gtIgnore   - [1xG] ignore flag for each gt
    #  dtIgnore   - [TxD] ignore flag for each dt at each IoU
    #
    # accumulate(): accumulates the per-image, per-category evaluation
    # results in "evalImgs" into the dictionary "eval" with fields:
    #  params     - parameters used for evaluation
    #  date       - date evaluation was performed
    #  counts     - [T,R,K,A,M] parameter dimensions (see above)
    #  precision  - [TxRxKxAxM] precision for every evaluation setting
    #  recall     - [TxKxAxM] max recall for every evaluation setting
    # Note: precision and recall==-1 for settings with no gt objects.
    #
    # See also coco, mask, pycocoDemo, pycocoEvalDemo
    #
    # Microsoft COCO Toolbox.      version 2.0
    # Data, paper, and tutorials available at:  http://mscoco.org/
    # Code written by Piotr Dollar and Tsung-Yi Lin, 2015.
    # Licensed under the Simplified BSD License [see coco/license.txt]
    def __init__(self, annotations=None, results=None, numCats=9, iouType='mAP', nproc=4):
        '''
        Initialize CocoEval using coco APIs for gt and dt
        :param sodaaGt: coco object with ground truth annotations
        :param sodaaDt: coco object with detection results
        :return: None
        '''
        if not iouType:
            print('iouType not specified. use default iouType: mAP')
        self.annotations = annotations  # ground truth
        self.results = results  # detections
        self.numCats = numCats
        self.nproc = nproc
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results [KxAxI] elements
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = SODAAParams(iouType=iouType)  # parameters
        self._paramsEval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        # TODO: get ids
        if self.annotations is not None:
            self._getImgAndCatIds()

    def _getImgAndCatIds(self):
        self.params.imgIds = [i for i, _ in enumerate(self.annotations)]
        self.params.catIds = [i for i in range(self.numCats)]

    def _prepare(self):
        '''
        Prepare ._gts and ._dts for evaluation based on params
        :return: None
        '''
        p = self.params
        if p.useCats:
            # TODO: we do not specify the area sue to no-area split so far.
            gts = list()
            insId = 0
            for i, imgAnn in enumerate(self.annotations):
                for j in range(len(imgAnn['labels'])):
                    gt = dict(
                        bbox = imgAnn['bboxes'][j],
                        area = imgAnn['bboxes'][j][2] * imgAnn['bboxes'][j][3],
                        category_id = imgAnn['labels'][j],
                        image_id = i,
                        id = insId,
                        ignore = 0  # no ignore
                    )
                    gts.append(gt)
                    insId += 1

            dts = list()
            insId = 0
            for i, imgRes in enumerate(self.results):
                for j, catRes in enumerate(imgRes):
                    if len(catRes) == 0:
                        continue
                    bboxes, scores = catRes[:, :5], catRes[:, -1]
                    for k in range(len(scores)):
                        dt = dict(
                            image_id = i,
                            bbox = bboxes[k],
                            score = scores[k],
                            category_id = j,
                            id = insId,
                            area = bboxes[k][2] * bboxes[k][3]
                        )
                        dts.append(dt)
                        insId += 1
        else:
            # TODO: add class-agnostic evaluation codes
            pass

        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt['image_id'], gt['category_id']].append(gt)
        for dt in dts:
            self._dts[dt['image_id'], dt['category_id']].append(dt)
        self.evalImgs = defaultdict(
            list)  # per-image per-category evaluation results
        self.eval = {}  # accumulated evaluation results

    def evaluate(self):
        if mp.get_start_method(allow_none=True) != 'spawn':
            print("INFO: Multiprocessing start method is not 'spawn'. Attempting to set it...")
            try:
                mp.set_start_method('spawn', force=True)
                print("SUCCESS: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                print(f"WARNING: Could not set start method to 'spawn': {e}")
                print(
                    "WARNING: This may lead to CUDA errors. Please consider setting it at the start of your main script.")

        '''
        Run per image evaluation on given images and store results
        '''
        p = self.params
        print('Evaluate annotation type *{}*'.format(p.iouType))

        self._prepare()
        catIds = p.catIds if p.useCats else [-1]

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            num_gpus = torch.cuda.device_count()
            devices_list = [f'cuda:{i}' for i in range(num_gpus)]
        else:
            devices_list = ['cpu']

        print(f"Found {len(devices_list)} device(s): {devices_list}. Using {self.nproc} processes.")

        print('Calculating IoUs...')
        tic = time.time()

        img2catLst = [(imgId, catId) for imgId in p.imgIds for catId in catIds]
        params_dict = {'maxDets': p.maxDets}

        worker_args = [(imgId, catId, self._gts, self._dts, params_dict)
                       for imgId, catId in img2catLst]

        num_tasks = len(worker_args)

        if self.nproc > 1:
            with Manager() as manager:
                counter = manager.Value('i', 0)
                init_args = (counter, devices_list)

                chunksize = (num_tasks + self.nproc - 1) // self.nproc
                if chunksize == 0: chunksize = 1

                print(f"Total tasks: {num_tasks}, Processes: {self.nproc}, Chunksize: {chunksize}")
                print('Calculating IoUs...')

                with Pool(self.nproc, initializer=init_worker_with_counter, initargs=init_args) as pool:
                    pbar = tqdm(pool.imap_unordered(top_level_compute_iou_worker, worker_args, chunksize=chunksize),
                                total=num_tasks,
                                desc="IoU Calculation Progress")
                    results = [r for r in pbar]

        else:
            global worker_device
            worker_device = devices_list[0]
            print('Calculating IoUs...')
            results = [top_level_compute_iou_worker(arg) for arg in tqdm(worker_args, desc="IoU Calculation Progress")]

        self.ious = {(r[0], r[1]): r[2] for r in results}

        toc = time.time()
        print('IoU calculation Done (t={:0.2f}s).'.format(toc - tic))

        print('Running per image evaluation...')
        tic = time.time()
        maxDet = p.maxDets[-1]

        contents = [self.evaluateImg(imgId, catId, areaRng, maxDet)
                    for catId in catIds
                    for areaRng in p.areaRng
                    for imgId in p.imgIds]

        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))
        self.evalImgs = contents
        self._paramsEval = copy.deepcopy(self.params)

    def evaluateImg(self, imgId, catId, aRng, maxDet):
        '''
        Perform evaluation for single category and image using vectorized logic.
        :return: dict (single image results)
        '''
        p = self.params
        if p.useCats:
            gt = self._gts[imgId, catId]
            dt = self._dts[imgId, catId]
        else:
            gt = [_ for cId in p.catIds for _ in self._gts[imgId, cId]]
            dt = [_ for cId in p.catIds for _ in self._dts[imgId, cId]]

        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g['ignore'] or (g['area'] < aRng[0] or g['area'] > aRng[1]):
                g['_ignore'] = 1
            else:
                g['_ignore'] = 0

        gtind = np.argsort([g['_ignore'] for g in gt], kind='mergesort')
        gt = [gt[i] for i in gtind]
        gtIg = np.array([g['_ignore'] for g in gt])  # gt ignore flags

        dtind = np.argsort([-d['score'] for d in dt], kind='mergesort')
        dt = [dt[i] for i in dtind[0:maxDet]]
        dtScores = np.array([d['score'] for d in dt])

        iscrowd = [0 for o in gt]

        T = len(p.iouThrs)
        G = len(gt)
        D = len(dt)

        gtm = np.zeros((T, G))  # ground truth matches
        dtm = np.zeros((T, D))  # detection matches
        dtIg = np.zeros((T, D))  # detection ignore

        if D > 0 and G > 0:
            ious = self.ious[imgId, catId][dtind[0:maxDet], :][:, gtind]

            gt_idx_for_dt = ious.argmax(axis=1)  # (D,) a gt index for each dt
            gt_iou_for_dt = ious.max(axis=1)  # (D,) the iou value

            for tind, t in enumerate(p.iouThrs):
                for dind, d in enumerate(dt):
                    iou = gt_iou_for_dt[dind]
                    gind = gt_idx_for_dt[dind]

                    if iou < t:
                        continue

                    if gtm[tind, gind] > 0:
                        continue

                    if iscrowd[gind]:
                        continue

                    gtm[tind, gind] = d['id']
                    dtm[tind, dind] = gt[gind]['id']
                    dtIg[tind, dind] = gtIg[gind]

        a = np.array([d['area'] < aRng[0] or d['area'] > aRng[1] for d in dt]).reshape((1, D))
        dtIg = np.logical_or(dtIg, np.logical_and(dtm == 0, np.repeat(a, T, 0)))

        # store results
        return {
            'image_id': imgId,
            'category_id': catId,
            'aRng': aRng,
            'maxDet': maxDet,
            'dtIds': [d['id'] for d in dt],
            'gtIds': [g['id'] for g in gt],
            'dtMatches': dtm,
            'gtMatches': gtm,
            'dtScores': [d['score'] for d in dt],
            'gtIgnore': gtIg,
            'dtIgnore': dtIg,
        }

    def accumulate(self, p=None):
        '''
        Accumulate per image evaluation results and store the result in
        self.eval

        :param p: input params for evaluation
        :return: None
        '''
        print('Accumulating evaluation results...')
        tic = time.time()
        if not self.evalImgs:
            print('Please run evaluate() first')
        # allows input customized parameters
        if p is None:
            p = self.params
        p.catIds = p.catIds if p.useCats == 1 else [-1]
        T = len(p.iouThrs)
        R = len(p.recThrs)
        K = len(p.catIds) if p.useCats else 1
        A = len(p.areaRng)
        M = len(p.maxDets)
        precision = -np.ones(
            (T, R, K, A, M))  # -1 for the precision of absent categories
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._paramsEval
        catIds = _pe.catIds if _pe.useCats else [-1]
        setK = set(catIds)
        setA = set(map(tuple, _pe.areaRng))
        setM = set(_pe.maxDets)
        setI = set(_pe.imgIds)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.catIds) if k in setK]
        m_list = [m for n, m in enumerate(p.maxDets) if m in setM]
        a_list = [
            n for n, a in enumerate(map(lambda x: tuple(x), p.areaRng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.imgIds) if i in setI]
        I0 = len(_pe.imgIds)
        A0 = len(_pe.areaRng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, maxDet in enumerate(m_list):
                    E = [self.evalImgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e['dtScores'][0:maxDet] for e in E])

                    # different sorting method generates slightly different
                    # results. mergesort is used to be consistent as Matlab
                    # implementation.
                    inds = np.argsort(-dtScores, kind='mergesort')
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e['dtMatches'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                          inds]
                    dtIg = np.concatenate(
                        [e['dtIgnore'][:, 0:maxDet] for e in E], axis=1)[:,
                                                                         inds]
                    gtIg = np.concatenate([e['gtIgnore'] for e in E])
                    npig = np.count_nonzero(gtIg == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dtIg))
                    fps = np.logical_and(np.logical_not(dtm),
                                         np.logical_not(dtIg))

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R, ))
                        ss = np.zeros((R, ))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for
                        # accessing elements use python array gets significant
                        # speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.recThrs, side='left')
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except:  # noqa: E722
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            'params': p,
            'counts': [T, R, K, A, M],
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'precision': precision,
            'recall': recall,
            'scores': scores,
        }
        toc = time.time()
        print('DONE (t={:0.2f}s).'.format(toc - tic))

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter
        setting
        '''
        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = '{:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'  # noqa: E501
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [
                i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng
            ]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            txtPth = "./work_dirs/evalRes.txt"
            txtFile = open(txtPth, 'a+')
            txtFile.writelines(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                                           mean_s))
            txtFile.writelines('\n')
            txtFile.close()
            print(
                iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets,
                            mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((12, ))
            # AP metric
            #
            stats[0] = _summarize(1,
                                  areaRng='Small',
                                  maxDets=self.params.maxDets[0])
            stats[1] = _summarize(1,
                                  iouThr=.50,
                                  areaRng='Small',
                                  maxDets=self.params.maxDets[0])
            stats[2] = _summarize(1,
                                  iouThr=.75,
                                  areaRng='Small',
                                  maxDets=self.params.maxDets[0])
            stats[3] = _summarize(1,
                                  areaRng='eS',
                                  maxDets=self.params.maxDets[0])
            stats[4] = _summarize(1,
                                  areaRng='rS',
                                  maxDets=self.params.maxDets[0])
            stats[5] = _summarize(1,
                                  areaRng='gS',
                                  maxDets=self.params.maxDets[0])
            stats[6] = _summarize(1,
                                  areaRng='Normal',
                                  maxDets=self.params.maxDets[0])

            # AR metric
            stats[7] = _summarize(0,
                                   areaRng='Small',
                                   maxDets=self.params.maxDets[0])
            stats[8] = _summarize(0,
                                   areaRng='eS',
                                   maxDets=self.params.maxDets[0])
            stats[9] = _summarize(0,
                                   areaRng='rS',
                                   maxDets=self.params.maxDets[0])
            stats[10] = _summarize(0,
                                   areaRng='gS',
                                   maxDets=self.params.maxDets[0])
            stats[11] = _summarize(0,
                                   areaRng='Normal',
                                   maxDets=self.params.maxDets[0])
            return stats


        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'mAP':
            summarize = _summarizeDets
        else:
            raise Exception('unknown iouType for iou computation')
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class SODAAParams:
    '''
    Params for coco evaluation api
    '''
    def __init__(self, iouType='mAP'):
        if iouType == 'mAP':
            self.setDetParams()
        else:
            raise Exception('iouType not supported')
        self.iouType = iouType

    def setDetParams(self):
        self.imgIds = []
        self.catIds = []
        # np.arange causes trouble.  the data point on arange is slightly
        # larger than the true value
        self.iouThrs = np.linspace(.5,
                                   0.95,
                                   int(np.round((0.95 - .5) / .05)) + 1,
                                   endpoint=True)
        self.recThrs = np.linspace(.0,
                                   1.00,
                                   int(np.round((1.00 - .0) / .01)) + 1,
                                   endpoint=True)
        # TODO: ensure
        self.maxDets = [20000]
        self.areaRng = [[0 ** 2, 32 ** 2], [0 ** 2, 12 ** 2], [12 ** 2, 20 ** 2],
                        [20 ** 2, 32 ** 2], [32 ** 2, 40 * 50]]
        self.areaRngLbl = ['Small', 'eS', 'rS', 'gS', 'Normal']
        self.useCats = 1

