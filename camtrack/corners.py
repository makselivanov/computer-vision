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
    MAX_CORNERS = 200
    MIN_DISTANCE = 15
    BLOCK_SIZE = 11 # 11

    image_0 = frame_sequence[0]
    norm_image_0 = cv2.normalize(image_0, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    corners = cv2.goodFeaturesToTrack(
        norm_image_0,
        maxCorners=MAX_CORNERS,
        qualityLevel=0.01,
        minDistance=MIN_DISTANCE,
        blockSize=BLOCK_SIZE,
    )
    # corners = FrameCorners(
    #    np.array([0]),
    #    np.array([[0, 0]]),
    #    np.array([55])
    # )
    corners = corners.reshape(-1, 2)
    index = corners.shape[0]
    corners = FrameCorners(np.arange(corners.shape[0]), corners, np.ones(corners.shape[0]) * 15)
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
            winSize=(BLOCK_SIZE, BLOCK_SIZE),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001),
            minEigThreshold=0.001)
        mask = np.ones_like(norm_image_1)
        for i, point in enumerate(p1):
            mask = cv2.circle(mask, center=(int(point[0]), int(point[1])), radius=MIN_DISTANCE, color=0, thickness=cv2.FILLED)
        pts = cv2.goodFeaturesToTrack(
            norm_image_1,
            maxCorners=MAX_CORNERS - st[st == 0].shape[0], # needs st[st == 1]
            qualityLevel=0.01,
            minDistance=MIN_DISTANCE,
            mask=mask,
            blockSize=BLOCK_SIZE,
        )
        pts = pts.reshape(-1, 2)
        j = 0
        newid = []
        newcorners = []
        for i, b in enumerate(st):
            if b == 0:
                if j < pts.shape[0]:
                    newid.append(index)
                    index += 1
                    newcorners.append(pts[j])
                    j += 1
            else:
                newid.append(corners.ids[i, 0])
                newcorners.append(p1[i])
        newcorners = np.array(newcorners)
        newid = np.array(newid)
        corners = FrameCorners(newid, newcorners, np.ones(newcorners.shape[0]) * 15)
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
