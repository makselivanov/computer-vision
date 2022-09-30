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
    create_cli
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


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    image_0 = frame_sequence[0]
    corners = cv2.goodFeaturesToTrack(
        image_0,
        maxCorners=50,
        qualityLevel=0.05,
        minDistance=10,
        #mask=np.ones_like(image_0).astype(np.uint8),
        blockSize=5,
    )
    #mask = np.zeros_like(image_0)
    # corners = FrameCorners(
    #    np.array([0]),
    #    np.array([[0, 0]]),
    #    np.array([55])
    # )
    index = corners.shape[0]
    corners = FrameCorners(np.arange(corners.shape[0]), corners, np.ones(corners.shape[0]) * 15)
    builder.set_corners_at_frame(0, corners)
    for frame, image_1 in enumerate(frame_sequence[1:], 1):

        builder.set_corners_at_frame(frame, corners)
        mask = np.zeros_like(image_0).astype(np.uint8)
        for point in corners.points:
            mask = cv2.circle(mask, center=(int(point[0]), int(point[1])), radius=30, color=1, thickness=cv2.FILLED)

        nextPts = cv2.goodFeaturesToTrack(
            image_0,
            maxCorners=50,
            qualityLevel=0.05,
            minDistance=10,
            mask=mask,
            blockSize=5,
        )
        if nextPts.shape[0] < 50:
            nextPts = None
        corners2, st, err = cv2.calcOpticalFlowPyrLK(
            prevImg=image_0.astype(np.uint8),
            nextImg=image_1.astype(np.uint8),
            prevPts=corners.points,
            nextPts=nextPts,
            winSize=(5, 5),
            flags = cv2.OPTFLOW_LK_GET_MIN_EIGENVALS | ((cv2.OPTFLOW_USE_INITIAL_FLOW) if nextPts is not None else 0)
        )
        newid = corners._ids.copy()
        for i, b in enumerate(st):
            if not b:
                newid[i] = index
                index += 1
        corners = FrameCorners(newid, corners2, corners.sizes)
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
