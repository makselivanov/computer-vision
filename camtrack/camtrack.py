#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import numpy as np
import sortednp as snp
import cv2
from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from scipy.spatial.transform import Rotation
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    triangulate_correspondences,
    Correspondences, TriangulationParameters
)


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()
    params = TriangulationParameters(1, 1, 0)

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    frame_count = len(corner_storage)
    is_camera_found = [False] * frame_count
    view_mats = [pose_to_view_mat3x4(known_view_1[1])] * frame_count
    is_camera_found[known_view_1[0]] = True
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])
    is_camera_found[known_view_2[0]] = True

    point_cloud_builder = PointCloudBuilder()
    corners_1 = corner_storage[known_view_1[0]]
    corners_2 = corner_storage[known_view_2[0]]

    ids, (idx_1, idx_2) = snp.intersect(corners_1.ids.flatten(), corners_2.ids.flatten(),
                                        indices=True)
    corr = Correspondences(ids, corners_1.points[idx_1], corners_2.points[idx_2])

    points3d, ids3d, error = triangulate_correspondences(
        corr,
        view_mats[known_view_1[0]], view_mats[known_view_2[0]],
        intrinsic_mat,
        params
    )

    point_cloud_builder.add_points(ids3d, points3d)

    loop = 2
    while loop < frame_count:
        new_ids3d, new_points3d, _ = point_cloud_builder.build_point_cloud()
        retval = False
        confidence = 1
        index = -1
        while not retval:
            confidence -= 0.01
            if confidence < 0:
                raise ValueError("Can't find confidence result")
            for index in range(frame_count):
                if not is_camera_found[index]:
                    cur_corner = corner_storage[index]
                    cur_ids, (cur_idx_1, cur_idx_2) = snp.intersect(cur_corner.ids.flatten(),
                                                                    new_ids3d.flatten(),
                                                                    indices=True)
                    if len(cur_ids) >= 4:
                        retval, rvec, tvec, inliners = cv2.solvePnPRansac(objectPoints=new_points3d[cur_idx_2],
                                                                          imagePoints=cur_corner.points[cur_idx_1],
                                                                          cameraMatrix=intrinsic_mat,
                                                                          distCoeffs=None,
                                                                          reprojectionError=1,
                                                                          confidence=confidence,
                                                                          iterationsCount=300
                                                                          )
                        if retval:
                            break
        is_camera_found[index] = True
        ratmat = Rotation.from_rotvec(rvec.reshape(-1)).as_matrix().T
        transp = -ratmat @ tvec
        view_mats[index] = pose_to_view_mat3x4(Pose(ratmat, transp))
        cur_corner = corner_storage[index]
        for other in range(frame_count):
            other_corner = corner_storage[other]
            if other != index and is_camera_found[other]:
                cur_ids, (cur_idx_1, cur_idx_2) = snp.intersect(cur_corner.ids.flatten(),
                                                                other_corner.ids.flatten(),
                                                                indices=True)
                cur_corr = Correspondences(cur_ids, cur_corner.points[cur_idx_1], other_corner.points[cur_idx_2])
                new_points3d, new_ids3d, error = triangulate_correspondences(cur_corr,
                                                                             view_mats[index],
                                                                             view_mats[other],
                                                                             intrinsic_mat, params)
                if error < params.max_reprojection_error:
                    point_cloud_builder.update_points(new_ids3d, new_points3d)
        loop += 1

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
