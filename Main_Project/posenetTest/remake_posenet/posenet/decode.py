import numpy as np
from posenet.constants import *

def traverse_to_targ_keypoint():
    temp =1


# 전처리된 데이터로부터 부위별 좌표를 정리해서 받아오는 함수
def decode_pose(
        root_score, root_id, root_image_coord,
        scores, offsets, output_stride,
        displacements_fwd, displascements_bwd
    ):

    num_parts = scores.shape[2]
    num_edges = len(PARENT_CHILD_TUPLES)

    #변수들 초기화
    instance_keypoint_scores = np.zeros(num_parts) # 확률
    instance_keypoint_coords = np.zeros((num_parts, 2)) # 좌표
    instance_keypoint_id = np.zeros(num_parts) # 부위 코드

    #데이터 입력
    instance_keypoint_scores[root_id] = root_score # 연산 확률
    instance_keypoint_coords[root_id] = root_image_coord #연산 좌표
    instance_keypoint_id[root_id] = root_id # 부위 코드

    #그래프와 유사한 구조로 저장
    for edge in reversed(range(num_edges)):
        target_keypoint_id, source_keypoint_id = PARENT_CHILD_TUPLES[edge]
        if (instance_keypoint_scores[source_keypoint_id] > 0.0 and 
                instance_keypoint_scores[target_keypoint_id] == 0.0):
            score, coords = traverse_to_targ_keypoint(
                edge,
                instance_keypoint_coords[source_keypoint_id],
                target_keypoint_id,
                score, offsets, output_stride, displascements_bwd
                )
            
            #여기서부터 파일 추가
            instance_keypoint_scores[target_keypoint_id] = score
            instance_keypoint_coords[target_keypoint_id] = coords


            

    