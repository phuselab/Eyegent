# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import colorsys
import logging
import math
import random
from enum import Enum, unique
import cv2
import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplfigure
import numpy as np
import pycocotools.mask as mask_util
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from .color_map import random_color
logger = logging.getLogger(__name__)
__all__ = ['ColorMode', 'VisImage', 'Visualizer']
_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)
_KEYPOINT_THRESHOLD = 0.05

@unique
class ColorMode(Enum):
    IMAGE = 0
    SEGMENTATION = 1
    IMAGE_BW = 2

class GenericMask:

    def __init__(self, mask_or_polygons, height, width):
        self._mask = self._polygons = self._has_holes = None
        self.height = height
        self.width = width
        m = mask_or_polygons
        if isinstance(m, dict):
            assert 'counts' in m and 'size' in m
            if isinstance(m['counts'], list):
                h, w = m['size']
                assert h == height and w == width
                m = mask_util.frPyObjects(m, h, w)
            self._mask = mask_util.decode(m)[:, :]
            return
        if isinstance(m, list):
            self._polygons = [np.asarray(x).reshape(-1) for x in m]
            return
        if isinstance(m, np.ndarray):
            assert m.shape[1] != 2, m.shape
            assert m.shape == (height, width), f'mask shape: {m.shape}, target dims: {height}, {width}'
            self._mask = m.astype('uint8')
            return
        raise ValueError("GenericMask cannot handle object {} of type '{}'".format(m, type(m)))

    @property
    def mask(self):
        if self._mask is None:
            self._mask = self.polygons_to_mask(self._polygons)
        return self._mask

    @property
    def polygons(self):
        if self._polygons is None:
            self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
        return self._polygons

    @property
    def has_holes(self):
        if self._has_holes is None:
            if self._mask is not None:
                self._polygons, self._has_holes = self.mask_to_polygons(self._mask)
            else:
                self._has_holes = False
        return self._has_holes

    def mask_to_polygons(self, mask):
        mask = np.ascontiguousarray(mask)
        res = cv2.findContours(mask.astype('uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:
            return ([], False)
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        res = [x + 0.5 for x in res if len(x) >= 6]
        return (res, has_holes)

    def polygons_to_mask(self, polygons):
        rle = mask_util.frPyObjects(polygons, self.height, self.width)
        rle = mask_util.merge(rle)
        return mask_util.decode(rle)[:, :]

    def area(self):
        return self.mask.sum()

    def bbox(self):
        p = mask_util.frPyObjects(self.polygons, self.height, self.width)
        p = mask_util.merge(p)
        bbox = mask_util.toBbox(p)
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        return bbox

class VisImage:

    def __init__(self, img, scale=1.0):
        self.img = img
        self.scale = scale
        self.width, self.height = (img.shape[1], img.shape[0])
        self._setup_figure(img)

    def _setup_figure(self, img):
        fig = mplfigure.Figure(frameon=False)
        self.dpi = fig.get_dpi()
        fig.set_size_inches((self.width * self.scale + 0.01) / self.dpi, (self.height * self.scale + 0.01) / self.dpi)
        self.canvas = FigureCanvasAgg(fig)
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis('off')
        self.fig = fig
        self.ax = ax
        self.reset_image(img)

    def reset_image(self, img):
        img = img.astype('uint8')
        self.ax.imshow(img, extent=(0, self.width, self.height, 0), interpolation='nearest')

    def save(self, filepath):
        self.fig.savefig(filepath)

    def get_image(self):
        canvas = self.canvas
        s, (width, height) = canvas.print_to_buffer()
        buffer = np.frombuffer(s, dtype='uint8')
        img_rgba = buffer.reshape(height, width, 4)
        rgb, alpha = np.split(img_rgba, [3], axis=2)
        return rgb.astype('uint8')

class Visualizer:

    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, font_size_multiplier=1.3, boarder_width_multiplier=1.5):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.boarder_width_multiplier = boarder_width_multiplier
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device('cpu')
        self._default_font_size = max(np.sqrt(self.output.height * self.output.width) // 60, 15 // scale) * font_size_multiplier
        self._instance_mode = instance_mode
        self.keypoint_threshold = _KEYPOINT_THRESHOLD
        import matplotlib.colors as mcolors
        css4_colors = mcolors.CSS4_COLORS
        self.color_proposals = [list(mcolors.hex2color(color)) for color in css4_colors.values()]

    def overlay_instances(self, *, boxes=None, labels=None, masks=None, keypoints=None, assigned_colors=None, binary_masks=None, alpha=0.5, label_mode='1'):
        num_instances = 0
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return (labels, [], [])
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])
        marks = []
        marks_position = []
        added_positions = set()
        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                self.draw_box(boxes[i], alpha=1, edge_color=color)
                if binary_masks is None:
                    mark = self._draw_number_in_box(boxes[i], i + 1, color=color, label_mode=label_mode)
                    marks.append(mark)
            if binary_masks is not None:
                mark, mask_position = self._draw_number_in_mask(binary_mask=binary_masks[i].astype('uint8'), text=i + 1, color=color, added_positions=added_positions, label_mode=label_mode)
                marks.append(mark)
                marks_position.append(mask_position)
                self.draw_binary_mask(binary_masks[i], color=color, edge_color=_OFF_WHITE, alpha=alpha)
            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=0)
        return (labels, marks, marks_position)

    def mask_dims_from_binary(self, binary_mask):
        ind_y, ind_x = np.where(binary_mask == 1)
        min_ind_x = np.min(ind_x)
        max_ind_x = np.max(ind_x)
        min_ind_y = np.min(ind_y)
        max_ind_y = np.max(ind_y)
        return (max_ind_x - min_ind_x, max_ind_y - min_ind_y)

    def reposition_label(self, position, cur, binary_mask, move_count):
        img_width, img_height = (self.output.width, self.output.height)
        mask_width, mask_height = self.mask_dims_from_binary(binary_mask)
        mask_width_limit, mask_height_limit = (25, 25)
        location_diff_threshold = 15
        x_boundry_limit, y_boundry_limit = (20, 20)
        offset_x = 15
        offset_y = 15
        x1, y1 = position
        if mask_width < mask_width_limit and mask_height < mask_height_limit and (move_count == 0):
            move_x = offset_x if offset_x + x1 < img_width else -offset_x
            move_y = offset_y if offset_y + y1 < img_height else -offset_y
            return (True, move_x, move_y)
        for x2, y2 in cur:
            if abs(x1 - x2) + abs(y1 - y2) < location_diff_threshold:
                move_x = offset_x if x1 >= x2 else -offset_x
                move_y = offset_y if y1 >= y2 else -offset_y
                move_x = 0 if x1 + move_x > img_width - x_boundry_limit or x1 + move_x < x_boundry_limit else move_x
                move_y = 0 if y1 + move_y > img_height - y_boundry_limit or y1 + move_y < y_boundry_limit else move_y
                return (True, move_x, move_y)
        return (False, 0, 0)

    def locate_label_position(self, original_position, added_positions, binary_mask):
        if added_positions is None or binary_mask is None:
            return original_position
        x, y = original_position
        move_count = 0
        reposition, x_move, y_move = self.reposition_label((x, y), added_positions, binary_mask, move_count)
        while reposition and move_count < 10:
            x += x_move
            y += y_move
            move_count += 1
            reposition, x_move, y_move = self.reposition_label((x, y), added_positions, binary_mask, move_count)
        added_positions.add((x, y))
        return (x, y)

    def draw_text(self, text, position, added_positions=None, binary_mask=None, *, font_size=None, color='g', horizontal_alignment='center', rotation=0):
        if not font_size:
            font_size = self._default_font_size
        color = np.maximum(list(mplc.to_rgb(color)), 0.15)
        color[np.argmax(color)] = max(0.8, np.max(color))

        def contrasting_color(rgb):
            R, G, B = rgb
            Y = 0.299 * R + 0.587 * G + 0.114 * B
            return 'black' if Y > 128 else 'white'
        bbox_background = contrasting_color(color * 255)
        x, y = self.locate_label_position(original_position=position, added_positions=added_positions, binary_mask=binary_mask)
        self.output.ax.text(x, y, text, size=font_size * self.output.scale, family='sans-serif', bbox={'facecolor': bbox_background, 'alpha': 0.8, 'pad': 0.7, 'edgecolor': 'none'}, verticalalignment='top', horizontalalignment=horizontal_alignment, color=color, zorder=10, rotation=rotation)
        return self.output

    def draw_box(self, box_coord, alpha=0.5, edge_color='g', line_style='-'):
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0
        linewidth = max(self._default_font_size / 12, 1) * self.boarder_width_multiplier
        self.output.ax.add_patch(mpl.patches.Rectangle((x0, y0), width, height, fill=False, edgecolor=edge_color, linewidth=linewidth * self.output.scale, alpha=alpha, linestyle=line_style))
        return self.output

    def draw_binary_mask(self, binary_mask, color=None, *, edge_color=None, text=None, alpha=0.7, area_threshold=10):
        if color is None:
            color = random_color(rgb=True, maximum=1)
        color = mplc.to_rgb(color)
        has_valid_segment = False
        binary_mask = binary_mask.astype('uint8')
        mask = GenericMask(binary_mask, self.output.height, self.output.width)
        shape2d = (binary_mask.shape[0], binary_mask.shape[1])
        if not mask.has_holes:
            for segment in mask.polygons:
                area = mask_util.area(mask_util.frPyObjects([segment], shape2d[0], shape2d[1]))
                if area < (area_threshold or 0):
                    continue
                has_valid_segment = True
                segment = segment.reshape(-1, 2)
                self.draw_polygon(segment, color=color, edge_color=edge_color, alpha=alpha)
        else:
            rgba = np.zeros(shape2d + (4,), dtype='float32')
            rgba[:, :, :3] = color
            rgba[:, :, 3] = (mask.mask == 1).astype('float32') * alpha
            has_valid_segment = True
            self.output.ax.imshow(rgba, extent=(0, self.output.width, self.output.height, 0))
        if text is not None and has_valid_segment:
            lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
            self._draw_text_in_mask(binary_mask, text, lighter_color)
        return self.output

    def draw_polygon(self, segment, color, edge_color=None, alpha=0.5):
        if edge_color is None:
            if alpha > 0.8:
                edge_color = self._change_color_brightness(color, brightness_factor=-0.7)
            else:
                edge_color = color
        edge_color = mplc.to_rgb(edge_color) + (1,)
        polygon = mpl.patches.Polygon(segment, fill=True, facecolor=mplc.to_rgb(color) + (alpha,), edgecolor=edge_color, linewidth=max(self._default_font_size // 15 * self.output.scale, 1))
        self.output.ax.add_patch(polygon)
        return self.output

    def _change_color_brightness(self, color, brightness_factor):
        assert brightness_factor >= -1.0 and brightness_factor <= 1.0
        color = mplc.to_rgb(color)
        polygon_color = colorsys.rgb_to_hls(*mplc.to_rgb(color))
        modified_lightness = polygon_color[1] + brightness_factor * polygon_color[1]
        modified_lightness = 0.0 if modified_lightness < 0.0 else modified_lightness
        modified_lightness = 1.0 if modified_lightness > 1.0 else modified_lightness
        modified_color = colorsys.hls_to_rgb(polygon_color[0], modified_lightness, polygon_color[2])
        return modified_color

    def _convert_boxes(self, boxes):
        return np.asarray(boxes)

    def _convert_masks(self, masks_or_polygons):
        return [x if isinstance(x, GenericMask) else GenericMask(x, self.output.height, self.output.width) for x in masks_or_polygons]

    def _draw_number_in_box(self, box, text, color, label_mode='1'):
        x0, y0, x1, y1 = box
        text_pos = (x0, y0)
        horiz_align = 'left'
        instance_area = (y1 - y0) * (x1 - x0)
        if instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale or y1 - y0 < 40 * self.output.scale:
            if y1 >= self.output.height - 5:
                text_pos = (x1, y0)
            else:
                text_pos = (x0, y1)
        height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
        lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
        font_size = np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2) * 0.65 * self._default_font_size
        if label_mode == 'a':
            text = self.number_to_string(int(text))
        else:
            text = text
        self.draw_text(text, text_pos, color=lighter_color, horizontal_alignment=horiz_align, font_size=font_size)
        return str(text)

    @staticmethod
    def number_to_string(n):
        chars = []
        while n:
            n, remainder = divmod(n - 1, 26)
            chars.append(chr(97 + remainder))
        return ''.join(reversed(chars))

    def _draw_number_in_mask(self, binary_mask, text, color, added_positions=None, label_mode='1'):
        binary_mask = np.pad(binary_mask, ((1, 1), (1, 1)), 'constant')
        mask_dt = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 0)
        mask_dt = mask_dt[1:-1, 1:-1]
        max_dist = np.max(mask_dt)
        coords_y, coords_x = np.where(mask_dt == max_dist)
        if label_mode == 'a':
            text = self.number_to_string(int(text))
        else:
            text = text
        text_position = (coords_x[len(coords_x) // 2] + 2, coords_y[len(coords_y) // 2] - 6)
        self.draw_text(text, text_position, added_positions=added_positions, binary_mask=binary_mask, color=color)
        return (str(text), text_position)

    def _draw_text_in_mask(self, binary_mask, text, color):
        _num_cc, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, 8)
        if stats[1:, -1].size == 0:
            return
        largest_component_id = np.argmax(stats[1:, -1]) + 1
        for cid in range(1, _num_cc):
            if cid == largest_component_id or stats[cid, -1] > _LARGE_MASK_AREA_THRESH:
                center = np.median((cc_labels == cid).nonzero(), axis=1)[::-1]
                bottom = np.max((cc_labels == cid).nonzero(), axis=1)[::-1]
                center[1] = bottom[1] + 2
                self.draw_text(text, center, color=color)
