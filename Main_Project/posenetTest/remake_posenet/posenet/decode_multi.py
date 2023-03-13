#from posenet.decode import *
from posenet.constants import *
import time
import scipy.ndimage as ndi
import numpy as np
from posenet.decode import *


#predict한 값을 바탕으로 신체부위를 특정할 수 있도록 빌드하는 함수
def build_part_with_score(score_threshold, local_max_radius, scores):
    parts = []
    num_keypoints = scores.shape[2]
    lmd = 2 * local_max_radius + 1 # 3

    #NOTE
    #전체 score_array에서 작업을 수행하는 것 보다
    # score_array를 분할한 sub array에서 각각 작업하는 것이 더 빠른 것 같다


    for keypoint_id in range(num_keypoints):
        #전체에서 keypoint_id 부분만 2차원 배열로 뽑아옴
        kp_scores = scores[:, :, keypoint_id].copy()
        # 뽑아온 데이터에서 min_data는 모두 제거한다
        # NOTE 제거하지 않으면 predict에서 0에 근접한 값이 배열에 가득차있다.
        kp_scores[kp_scores < score_threshold] = 0
        #경계선을 필터링하여 정의
        max_vals = ndi.maximum_filter(kp_scores, size = lmd, mode='constant')
        #max_vals에서 0보다 크고 kp_scores과 같은 것은 모두 True, 이외는 False
        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)
        # np.nonzero() : 0이 아닌 어레이의 index를 반환
        max_loc_idx = max_loc.nonzero()

        #확률, 부위, x, y array(x, y배열은 어디다 쓰는 거지?)
        for y, x in zip(*max_loc_idx):
            parts.append((
                scores[y, x, keypoint_id],
                keypoint_id,
                np.array((y, x))
            ))
        
        #parts = [확률, 부위, xy array]
        return parts
    
def within_nms_radius_fast(pose_coords, squared_nms_radius, point):
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis = 1) <= squared_nms_radius)


def dcode_multiple_poses(
        scores,
        offsets,
        displacements_fwd,
        displacements_bwd,
        output_stride,
        max_pose_detections = 10,
        score_threshold = 0.5,
        nms_radius = 20, score = 0.5
        ):
    
    #포즈 분석 변수 초기화

    pose_count = 0 # count 
    # [0,0,0,0,0 ... ] 10개
    pose_score = np.zeros(max_pose_detections)
    # [[0,0,0,...], ... , [0,0,0,...]] 10 x 17개
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    # [[[0,0,0, ...], ... , [0,0,0,...]], ... ,[[0,0,0, ...], ... , [0,0,0,...]]] 10 x 17 x 2 개
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))

    squared_nms_radius = nms_radius ** 2 # 400

    #score_thres_hold = min_pose_score = 0.15
    #LOCAL_MAXIMUM_RADIUS = 1
    #scores = heatmap_result
    #part를 리턴시킴
    scored_parts = build_part_with_score(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    """
    scored_parts = [확률, 부위코드, XYarray]
    """
    #정렬
    scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)

    # heatmap의 가로 세로 길이
    height = scores.shape[0]
    width = scores.shape[1]
    # offset 세팅
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    # 연산결과 세팅
    displacements_fwd = displacements_fwd.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(height, width, 2, -1).swapaxes(2, 3)

    for root_score, root_id, root_coord in scored_parts:
        # 좌표평면에 지정
        root_image_coords = root_coord * output_stride + offsets[
            root_coord[0], root_coord[1] ,root_id
        ]

        #어디에 사용하는지 모르겠음. 거리측정?
        if within_nms_radius_fast(
            pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords
               ):
            continue

        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            scores, offsets, output_stride,
            displacements_fwd, displacements_bwd
        )








    


    





    



