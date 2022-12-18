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
    eye3x4,
    rodrigues_and_translation_to_view_mat3x4,
    Correspondences, TriangulationParameters
)


def inverse_transform(r: np.array, t: np.array):
    return r.inv(), -r.inv().apply(t)

def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    params = TriangulationParameters(1, 1, 0)

    # print("Container max id: ", corner_storage.max_corner_id())
    # print("Container len: ", len(corner_storage))

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)

    # print("rgb_sequence : ", len(rgb_sequence))

    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        CONFIDENCE = 0.90
        MAX_ITERS = 10 ** 4
        THRESHOLD_PX = 3.0
        THRESHOLD_HOMOGRAPH = 0.6
        THRESHOLD_ANGLE_DEGREE = 1

        params_opencv = dict(
            method=cv2.USAC_MAGSAC,
            #ransacReprojThreshold=THRESHOLD_PX,
            #threshold=THRESHOLD_PX,
            confidence=CONFIDENCE,
            maxIters=MAX_ITERS
        )

        # Lets for beginning we take first image and best image of all others
        known_view_1 = (0, view_mat3x4_to_pose(eye3x4()))

        max_verified_points = 0
        max_homogr_points = 0
        best_index_1 = -1
        best_index_2 = -1
        best_r, best_t = None, None
        assert (np.array_equal(sorted(corner_storage[0].ids), corner_storage[0].ids))
        for index_1 in range(0, len(corner_storage)):
            for index_2 in range(index_1 + 20, min(len(corner_storage), index_1 + 80)): # 20 here is min shift
                # Check Homography if too much
                ids, (idx_1, idx_2) = snp.intersect(corner_storage[index_1].ids.flatten(), corner_storage[index_2].ids.flatten(),
                                                    indices=True)
                points1 = corner_storage[index_1].points[idx_1]
                points2 = corner_storage[index_2].points[idx_2]
                matrix_homogr, mask_homogr = cv2.findHomography(points1, points2,
                                                                ransacReprojThreshold=THRESHOLD_PX,
                                                                **params_opencv)
                matrix, mask = cv2.findEssentialMat(points1, points2,
                                                    method=cv2.RANSAC,
                                                    cameraMatrix=intrinsic_mat,
                                                    threshold=THRESHOLD_PX,
                                                    prob=CONFIDENCE,
                                                    maxIters=MAX_ITERS)
                # Check angles is too smallt
                if mask_homogr.sum() > THRESHOLD_HOMOGRAPH * mask.sum():
                    #print("TOO MUCH homogr")
                    continue
                corr = Correspondences(np.arange(len(points1)), points1, points2)
                retval, r, t, inliers = cv2.recoverPose(E=matrix,
                                    points1=points1,
                                    points2=points2,
                                    cameraMatrix=intrinsic_mat)
                if max(abs(Rotation.from_matrix(r).as_euler("xyz", True))) < THRESHOLD_ANGLE_DEGREE:
                    continue
                if max_verified_points < retval:
                    max_verified_points = retval
                    max_homogr_points = mask_homogr.sum()
                    best_index_1 = index_1
                    best_index_2 = index_2
                    best_r, best_t = r, t
        print("intrinsic_mat is\n: ", intrinsic_mat)
        print("Best indexs is: ", best_index_1, best_index_2)
        print("Max good points: ", max_verified_points)
        print("Homogr points: ", max_homogr_points)
        print("R: \n", best_r, "\nt: \n", best_t)
        inv_r, inv_t = inverse_transform(Rotation.from_matrix(best_r), best_t.reshape(-1))
        known_view_2 = (best_index_2, Pose(inv_r.as_matrix(), inv_t))
        known_view_1 = (best_index_1, view_mat3x4_to_pose(eye3x4()))
        print("First image: \n", known_view_1)
        print("Second image: \n", known_view_2)
        # END FOR SEARCHING TWO IMAGE

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

        mn_index = -1
        max_len_cur_ids = 14
        max_rvec = None
        max_tvec = None
        max_retval = False
        while not max_retval:
            confidence -= 0.01
            if confidence < 0:
                print("Cur loop: ", loop)
                raise ValueError("Can't find confidence result")
            for index in range(frame_count):
                if not is_camera_found[index]:
                    cur_corner = corner_storage[index]
                    cur_ids, (cur_idx_1, cur_idx_2) = snp.intersect(cur_corner.ids.flatten(),
                                                                    new_ids3d.flatten(),
                                                                    indices=True)
                    if len(cur_ids) >= max_len_cur_ids:
                        retval, rvec, tvec, inliers = cv2.solvePnPRansac(objectPoints=new_points3d[cur_idx_2],
                                                                          imagePoints=cur_corner.points[cur_idx_1],
                                                                          cameraMatrix=intrinsic_mat,
                                                                          distCoeffs=None,
                                                                          reprojectionError=1,
                                                                          confidence=confidence,
                                                                          iterationsCount=300
                                                                          )
                        if retval:
                            max_retval = True
                            mn_index = index
                            max_len_cur_ids = len(cur_ids)
                            max_rvec = rvec
                            max_tvec = tvec
                            pass

                        #if retval:
                        #    break
        index = mn_index
        rvec = max_rvec
        tvec = max_tvec
        is_camera_found[index] = True
        ratmat, transp = inverse_transform(Rotation.from_rotvec(rvec.reshape(-1)), tvec.reshape(-1))
        # ratmat = Rotation.from_rotvec(rvec.reshape(-1)).as_matrix().T
        # transp = -ratmat @ tvec
        view_mats[index] = pose_to_view_mat3x4(Pose(ratmat.as_matrix(), transp))
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
