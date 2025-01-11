# Copyright (c) Open-CD. All rights reserved.
import os
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
import numpy as np
from mmengine.runner import Runner

from mmseg.engine import SegVisualizationHook
from mmseg.structures import SegDataSample
from opencd.registry import HOOKS
from opencd.visualization import CDLocalVisualizer


import cv2
import torch
import numpy as np

@HOOKS.register_module()
class CDVisualizationHook(SegVisualizationHook):
    """Change Detection Visualization Hook. Used to visualize validation and
    testing process prediction results. 

    Args:
        img_shape (tuple): if img_shape is given and `draw_on_from_to_img` is
            False, the original images will not be read.
        draw_on_from_to_img (bool): whether to draw semantic prediction results
            on the original images. If it is False, it means that drawing on
            the black board. Defaults to False.
    
    """
    def __init__(self,
                 img_shape: tuple = None,
                 draw_on_from_to_img: bool = False,
                 draw: bool = True,
                 interval: int = 1,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self.img_shape = img_shape
        self.draw_on_from_to_img = draw_on_from_to_img
        if self.draw_on_from_to_img:
            warnings.warn('`draw_on_from_to_img` works only in '
                          'semantic change detection.')
        self._visualizer: CDLocalVisualizer = \
            CDLocalVisualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        # for i, output in enumerate(outputs):
        #     seg_data_sample = output
        #     if hasattr(seg_data_sample, 'img_a_path'):
        #         print(f"Loaded image {i}: {seg_data_sample.img_a_path}")
        #     else:
        #         print(f"Sample {i} does not have img_a_path.")
        print(f"Batch {batch_idx} contains {len(outputs)} samples.")

        if self.draw is False or mode == 'train':
            return
        print("self.interval = ", self.interval)
        if self.every_n_inner_iters(batch_idx, self.interval):
            print(f"\n处理批次：{batch_idx}...")
            #image_num = 0
            for output in outputs:
                # print(output)
                # Sample data
                seg_data_sample = output
                img_name = osp.basename(seg_data_sample.img_a_path)
                # print(img_name)
                
                #image_num += 1
                #print(image_num)
#==============================================================================================================================
                # Extract data fields
                #gt_sem_seg_from = seg_data_sample.gt_sem_seg_from.data.cpu().numpy()
                # seg_logits = seg_data_sample.seg_logits.data.cpu().numpy()
                pred_sem_seg = seg_data_sample.pred_sem_seg.data.cpu().numpy()
                img_name = osp.basename(seg_data_sample.img_a_path)
                
                # Convert 1's in pred_sem_seg to 255
                # gt_sem_seg_from[gt_sem_seg_from != 0] = 255
                pred_sem_seg[pred_sem_seg != 0] = 255

                # 指定保存目录
                save_dir = "/home/iris/PycharmProjects/RSBuilding/result/mua4"
                os.makedirs(save_dir, exist_ok=True)

                # 保存结果
                save_path = osp.join(save_dir, img_name)
                print("保存结果到：", save_path)
                # Save images
                # cv2.imwrite("gt_sem_seg_from.png", gt_sem_seg_from.transpose(1, 2, 0))  # Assuming data is in HWC format
                # cv2.imwrite("seg_logits.png", seg_logits.transpose(1, 2, 0))
                cv2.imwrite(save_path, pred_sem_seg.transpose(1, 2, 0))
#==============================================================================================================================

                # #img_path = output.img_path[0]
                # img_path = output.img_a_path
                
                # img_from_to = []
                # window_name = osp.basename(img_path).split('.')[0]
                
                # if self.img_shape is not None:
                #     assert len(self.img_shape) == 3, \
                #         '`img_shape` should be (H, W, C)'
                    
                # else:
                #     img_bytes = fileio.get(
                #         img_path, backend_args=self.backend_args)
                #     img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                #     self.img_shape = img.shape
                
                # if self.draw_on_from_to_img:
                #     # for semantic change detection
                #     for _img_path in output.img_path:
                        
                #         _img_bytes = fileio.get(
                #             _img_path, backend_args=self.backend_args)
                #         _img = mmcv.imfrombytes(_img_bytes, channel_order='rgb')
                #         img_from_to.append(_img)
                    
                       
                # img = np.zeros(self.img_shape)
                # # print("=======window_name===========", window_name)
                # # print("=======img===========", img)
                # # print("=======img_from_to===========", output)
                # self._visualizer.add_datasample(
                #     window_name,
                #     img,
                #     img_from_to,
                #     data_sample=output,
                #     show=self.show,
                #     wait_time=self.wait_time,
                #     step=runner.iter,
                #     draw_gt=False)