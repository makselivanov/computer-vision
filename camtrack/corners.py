#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
from numba import njit

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli,
    filter_frame_corners
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def get_size_of_corners(image, corners, **kwargs):
    eigens = cv2.cornerMinEigenVal(np.uint8(image), **kwargs)
    max_eigen = np.max(eigens)
    return np.apply_along_axis(lambda point: eigens[int(point[1]), int(point[0])], 1, corners) / max_eigen * 25 + 10


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    MAX_CORNERS = 200 #200
    MIN_DISTANCE = 20 #15, 10 or 30
    QUALITY_LEVEL = 0.011
    BLOCK_SIZE = 10 # 9

    image_0 = frame_sequence[0]
    norm_image_0 = cv2.normalize(image_0, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    corners = cv2.goodFeaturesToTrack(
        norm_image_0,
        maxCorners=MAX_CORNERS,
        qualityLevel=QUALITY_LEVEL,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
    )

    corners = corners.reshape(-1, 2)
    index = corners.shape[0]
    corners = FrameCorners(np.arange(corners.shape[0]), corners,
                           get_size_of_corners(norm_image_0, corners, blockSize=BLOCK_SIZE, ksize=3)
                           )
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):
        p0 = corners.points
        norm_image_0 = cv2.normalize(image_0, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        norm_image_1 = cv2.normalize(image_1, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        p1, st, _err = cv2.calcOpticalFlowPyrLK(
            prevImg=norm_image_0,
            nextImg=norm_image_1,
            prevPts=p0,
            nextPts=None,
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            winSize=(15, 15),
            maxLevel=5,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            #minEigThreshold=0.00011) #0.0011
        )
        corners = FrameCorners(corners.ids, p1, corners.sizes)
        corners = filter_frame_corners(corners, (st == 1).reshape(-1))
        corners = filter_frame_corners(corners, np.apply_along_axis(
            lambda point: 0 <= point[0] < image_1.shape[1] and 0 <= point[1] < image_1.shape[0],
            1, corners.points
        ))
        mask = np.ones_like(norm_image_1)
        for i, p in enumerate(corners.points):
            y, x = int(p[1]), int(p[0])
            if mask[y, x] == 1:
                mask = cv2.circle(mask,
                                  center=(x, y),
                                  radius=MIN_DISTANCE,
                                  color=0,
                                  thickness=cv2.FILLED)
        pts = None
        if MAX_CORNERS - corners.points.shape[0] > 0:
            pts = cv2.goodFeaturesToTrack(
                norm_image_1,
                maxCorners=MAX_CORNERS - corners.points.shape[0],
                qualityLevel=QUALITY_LEVEL,
                minDistance=MIN_DISTANCE,
                mask=mask,
                blockSize=BLOCK_SIZE,
            )
        if pts is not None:
            pts = pts.reshape(-1, 2)
            corners = FrameCorners(
                np.concatenate([corners.ids.reshape(-1), np.arange(index, index + pts.shape[0])]),
                np.concatenate([corners.points, pts]),
                np.concatenate([corners.sizes.reshape(-1), get_size_of_corners(norm_image_1, pts, blockSize=BLOCK_SIZE, ksize=3)])
            )
            index += pts.shape[0]
        builder.set_corners_at_frame(frame, corners)
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
