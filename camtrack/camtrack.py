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
    Correspondences, TriangulationParameters,
    compute_reprojection_errors,
    build_correspondences
)


def inverse_transform(r: np.array, t: np.array):
    return r.inv(), -r.inv().apply(t)


def calc_triangulation_angles(camera_pose_1, camera_pose_2, points3d):
    rmat1 = camera_pose_1.r_mat
    t1 = camera_pose_1.t_vec
    rmat2 = camera_pose_2.r_mat
    t2 = camera_pose_2.t_vec
    vec1 = (rmat1 @ (2 * np.linalg.inv(rmat1) @ (points3d - t1).T)).T + t1 - points3d
    vec2 = (rmat2 @ (2 * np.linalg.inv(rmat2) @ (points3d - t2).T)).T + t2 - points3d
    angles = []
    for v1, v2 in zip(vec1, vec2):
        angles.append(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
    return np.array(angles)


def find_new_best_view(unknown_frames, known_frames,
                       corner_storage, ids3d, points3d,
                       view_mats, intrinsic_mat):
    ITERATIONS_COUNT = 200
    CONFIDENCE = 0.99
    REPROJECTION_ERROR = 8.
    ERROR_THRESHOLD = 10
    min_error = 100
    min_frame = None
    min_view = None

    for frame in unknown_frames:
        corners = corner_storage[frame]
        cur_ids, (cur_idx_1, cur_idx_2) = snp.intersect(corners.ids.flatten(), ids3d.flatten(), indices=True)
        retval, rvec, tvec, _ = cv2.solvePnPRansac(points3d[cur_idx_2], corners.points[cur_idx_1],
                                                         intrinsic_mat, None,
                                                         reprojectionError=REPROJECTION_ERROR,
                                                         confidence=CONFIDENCE,
                                                         iterationsCount=ITERATIONS_COUNT)
        if not retval:
            continue
        # rvec, tvec = inverse_transform(Rotation.from_rotvec(rvec.reshape(-1)), tvec.reshape(-1))
        # rvec = rvec.as_rotvec()
        view_mat = rodrigues_and_translation_to_view_mat3x4(rvec, tvec.reshape(-1, 1))
        error = compute_reprojection_errors(points3d[cur_idx_2], corners.points[cur_idx_1], intrinsic_mat @ view_mat).mean()
        if error < min_error:
            min_error = error
            min_frame = frame
            min_view = view_mat
    if min_error > ERROR_THRESHOLD:
        # TODO
        print("Too big error for new view ", min_error)
        pass
    return min_frame, min_view


params = TriangulationParameters(8., 0.5, 0.2)


def triangulate_new_points(new_frame, known_frames,
                           corner_storage, ids3d,
                           view_mats, intrinsic_mat, point_cloud_builder):
    ANGLE_THESHOLD = 5.
    curr_corners = corner_storage[new_frame]
    curr_view = view_mats[new_frame]
    for frame in known_frames:
        corners = corner_storage[frame]
        view = view_mats[frame]
        correspondences = build_correspondences(corners, curr_corners, ids3d)
        if len(correspondences.ids) == 0:
            continue
        new_points3d, new_ids3d, cos = triangulate_correspondences(correspondences, view, curr_view, intrinsic_mat, params)
        angle = np.arccos(cos) * 180 / np.pi
        if new_points3d is None or angle < ANGLE_THESHOLD:
            continue
        point_cloud_builder.update_points(new_ids3d, new_points3d)
        ids3d, _, _ = point_cloud_builder.build_point_cloud()


def find_best_init_frames(corner_storage, intrinsic_mat):
    POINTS_THRESHOLD = 20
    THRESHOLD_HOMOGRAPH = 0.7
    INLIERS_THRESHOLD = 5
    frame_count = len(corner_storage)
    max_angle = -1
    best_view_mat_1 = None
    best_view_mat_2 = None
    step = 10
    if frame_count < 150:
        step = 5
    if frame_count < 60:
        step = 1
    for frame_1 in range(0, frame_count, step):
        for frame_2 in range(frame_1 + step, frame_count, step):
            print("Current init frames: ", frame_1, frame_2)
            ids, (idx_1, idx_2) = snp.intersect(corner_storage[frame_1].ids.flatten(),
                                                corner_storage[frame_2].ids.flatten(),
                                                indices=True)
            points1 = corner_storage[frame_1].points[idx_1]
            points2 = corner_storage[frame_2].points[idx_2]
            if len(points1) < POINTS_THRESHOLD:
                continue
            matrix_homogr, mask_homogr = cv2.findHomography(points1, points2, cv2.RANSAC)
            matrix, mask = cv2.findEssentialMat(points1, points2, intrinsic_mat, cv2.RANSAC)

            if mask_homogr.sum() >= mask.sum() * THRESHOLD_HOMOGRAPH:
                continue
            retval, r, t, _ = cv2.recoverPose(matrix, points1, points2, intrinsic_mat, cv2.RANSAC)
            if retval < INLIERS_THRESHOLD:
                continue
            view_mat_1 = pose_to_view_mat3x4(Pose(np.eye(3), np.zeros(3)))
            r, t = inverse_transform(Rotation.from_matrix(r), t.reshape(-1))
            r = r.as_matrix()
            view_mat_2 = pose_to_view_mat3x4(Pose(r, t.reshape(-1)))
            corr = build_correspondences(corner_storage[frame_1], corner_storage[frame_2])
            if len(corr.ids) == 0:
                continue
            points_3d, ids_3d, cos = triangulate_correspondences(corr, view_mat_1, view_mat_2, intrinsic_mat, params)
            if len(ids_3d) == 0:
                continue
            angle = abs(np.arccos(cos))
            if angle > max_angle:
                max_angle = angle
                best_view_mat_1 = (frame_1, view_mat3x4_to_pose(view_mat_1))
                best_view_mat_2 = (frame_2, view_mat3x4_to_pose(view_mat_2))
    return best_view_mat_1, best_view_mat_2


def retriangulate(known_frames, corner_storage,
                  ids3d, points3d,
                  view_mats, intrinsic_mat):
    for (index, id) in enumerate(ids3d):
        cur_points2d = []
        cur_view_mats = []
        for frame in known_frames:
            corners = corner_storage[frame]
            if id in corners.ids.flatten():
                cur_view_mats.append(view_mats[frame])
                cur_points2d.append(corners.points[(corners.ids.flatten() == id)].reshape(-1))
        if len(cur_points2d) >= 5:
            eq = np.zeros(shape=(2 * len(cur_points2d), 4), dtype=float)
            for (i, (view_mat, point2d)) in enumerate(zip(cur_view_mats, cur_points2d)):
                proj = intrinsic_mat @ view_mat
                eq[2 * i] = proj[2] * point2d[0] - proj[0]
                eq[2 * i + 1] = proj[2] * point2d[1] - proj[1]
            new_point3d = np.linalg.lstsq(eq[:, :3], -eq[:, 3], rcond=None)[0]
            points3d[index] = new_point3d


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    frame_count = len(corner_storage)
    #print("Path: ", frame_sequence_path)
    print("Number of frames: ", frame_count)
    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    if known_view_1 is None or known_view_2 is None:
        known_view_1, known_view_2 = find_best_init_frames(corner_storage, intrinsic_mat)
    # END OF FIND INITIAL TWO IMAGE
    print("End of initialization")
    view_mats = [pose_to_view_mat3x4(known_view_1[1]) for _ in range(frame_count)]
    view_mats[known_view_2[0]] = pose_to_view_mat3x4(known_view_2[1])

    point_cloud_builder = PointCloudBuilder()
    corners_1 = corner_storage[known_view_1[0]]
    corners_2 = corner_storage[known_view_2[0]]

    corr = build_correspondences(corners_1, corners_2)

    unknown_frames = set(range(frame_count))
    unknown_frames.remove(known_view_1[0])
    unknown_frames.remove(known_view_2[0])
    known_frames = [known_view_1[0], known_view_2[0]]

    # TODO better triangulate current know 1 and know 2
    points3d, ids3d, error = triangulate_correspondences(
        corr,
        view_mats[known_view_1[0]], view_mats[known_view_2[0]],
        intrinsic_mat,
        params
    )

    point_cloud_builder.add_points(ids3d, points3d)
    for i in range(frame_count - 2):
        print("Already parsed frames: ", i + 2, " of ", frame_count)
        ids3d, points3d, _ = point_cloud_builder.build_point_cloud()
        new_frame, new_view = find_new_best_view(unknown_frames, known_frames,
                                                 corner_storage, ids3d, points3d,
                                                 view_mats, intrinsic_mat)
        view_mats[new_frame] = new_view
        unknown_frames.remove(new_frame)
        triangulate_new_points(new_frame, known_frames,
                               corner_storage, ids3d,
                               view_mats, intrinsic_mat, point_cloud_builder)
        known_frames.append(new_frame)

        if i % 2 == 0:
            retriangulate(known_frames, corner_storage, ids3d, points3d, view_mats, intrinsic_mat)
            pass

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        8.
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
