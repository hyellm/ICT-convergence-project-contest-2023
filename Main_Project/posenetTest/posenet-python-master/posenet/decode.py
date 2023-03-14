import numpy as np

from posenet.constants import *


# 주어진 edge_id와 source_keypoint를 사용하여 target_keypoint_id와 일치하는 키포인트를 찾는 함수 
def traverse_to_targ_keypoint(
        edge_id, source_keypoint, target_keypoint_id, scores, offsets, output_stride, displacements):
    height = scores.shape[0]
    width = scores.shape[1]

    # clip함수를 사용하여 source_keypoint_indices를 계산 (계산된 인덱스가 배열의 범위를 벗어나지 않도록 하기 위한 것)
    # np.clip = 지정된 범위내에서 잘라냄 (min보다 작으면 min / max보다 크면 max로 변환)
    source_keypoint_indices = np.clip(
        np.round(source_keypoint / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    # displaced_point = 현재 키포인트와 연결된 다음 키포인트로 이동한 점의 좌표
    displaced_point = source_keypoint + displacements[
        source_keypoint_indices[0], source_keypoint_indices[1], edge_id]

    # clip함수를 사용하여 displaced_point_indices를 계산 (다음 키포인트의 위치를 배열 인덱스 형태로 계산하기 위한 것)
    displaced_point_indices = np.clip(
        np.round(displaced_point / output_stride), a_min=0, a_max=[height - 1, width - 1]).astype(np.int32)

    # score = 현재 연결된 키포인트와 연결된 다음 키포인트에 대한 예측 점수
    score = scores[displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    # image_coord = 예측된 다음 키포인트의 이미지 상의 좌표 
    image_coord = displaced_point_indices * output_stride + offsets[
        displaced_point_indices[0], displaced_point_indices[1], target_keypoint_id]

    return score, image_coord

# 예측한 키포인트 관련 정보를 사용하여 이미지 상에서 실제 키포인트 좌표를 디코딩하는 함수 
def decode_pose(
        root_score, root_id, root_image_coord,
        scores,
        offsets,
        output_stride,
        displacements_fwd,
        displacements_bwd
):
    print('decod_pose function')
    num_parts = scores.shape[2]
    num_edges = len(PARENT_CHILD_TUPLES)

    # instance_keypoint_scores = 키포인트 각각의 신뢰도 값을 저장하는 배열
    # instance_keypoint_coords = 키포인트 각각의 좌표 값을 저장하는 2차원 배열 
    instance_keypoint_scores = np.zeros(num_parts)
    instance_keypoint_coords = np.zeros((num_parts, 2))
    instance_keypoint_scores[root_id] = root_score
    instance_keypoint_coords[root_id] = root_image_coord
    print('root score : ', root_score)
    print('root coords : ', root_image_coord)
    print('root id : ', root_id)


    # 이미지에서 키포인트 간의 관계를 연결하는 작업을 수행(역순으로 반복)
    for edge in reversed(range(num_edges)):

        # 배열에서 추출한 각 edge의 출발점과 도착점 설정
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]

        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            # traverse_to_targ_keypoint함수를 통해 현재 edege를 따라 도착점까지 탐색하면서 도착점의 좌표값과 신뢰도 값 추출
            score, coords = traverse_to_targ_keypoint(
                edge,instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_bwd)
<<<<<<< Updated upstream
            
            print('score : ', score)
            print('coords : ', coords)
            print('target_keypoint_id : ', target_keypoint_id)
            print('source_keypoint_id : ', source_keypoint_id)
=======

            # 추출한 도착점의 좌표값과 신뢰도 값을 배열에 저장
>>>>>>> Stashed changes
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    # 동작 방식은 위의 for문과 동일하나 순서대로 반복한다는 것이 차이점 (역순 X)
    for edge in range(num_edges):
        source_keypoint_id, target_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                scores, offsets, output_stride, displacements_fwd)
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords

    # 반환값은 추출한 도착점의 좌표값과 신뢰도 값이 담아져있는 배열
    return instance_keypoint_scores, instance_keypoint_coords
