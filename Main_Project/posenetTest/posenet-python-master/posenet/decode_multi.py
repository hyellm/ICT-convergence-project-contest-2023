# 

from posenet.decode import *
from posenet.constants import *
import time
import scipy.ndimage as ndi

<<<<<<< Updated upstream
# score가 무엇을 뜻하는걸까.....?

# point와 거리 계산하는 함수?
=======
# 주어진 keypoint_id와 해당 keypoint의 위치를 받아,
# poses에 있는 다른 인간 포즈들 중 squared_nms_radius 내에 있는지 확인하는 함수
>>>>>>> Stashed changes
def within_nms_radius(poses, squared_nms_radius, point, keypoint_id):

    # poses 배열의 각 요소는 튜플 형태로 되어있으며 그중 세번째 요소인 pose_coord의 값을 이용
    for _, _, pose_coord in poses:
        # pose_coord[keypoint_id]는 현재 처리 중인 키포인트의 좌표값을 의미
        # squared_nms_radius는 NMS에서 사용되는 반경 값의 제곱 
        if np.sum((pose_coord[keypoint_id] - point) ** 2) <= squared_nms_radius:
            return True
    return False

<<<<<<< Updated upstream
# 위의 함수와 비슷한 기능을 수행 (입력으로 들어가는 리스트가 다름)
=======
# numpy 함수를 이용하여 빠르게 인간 포즈들 중 squared_nms_radius 내에 있는지 확인하는 함수
>>>>>>> Stashed changes
def within_nms_radius_fast(pose_coords, squared_nms_radius, point):

    # pose_coords 배열은 모든 키포인트의 좌표값을 담고있음 (각 행은 하나의 키포인트)
    if not pose_coords.shape[0]:
        return False
    return np.any(np.sum((pose_coords - point) ** 2, axis=1) <= squared_nms_radius)

<<<<<<< Updated upstream
# 새로 계산된 keypoint들의 score를 반환하기 위해 사용되는 함수 
# keypoint_scores와 keypoint_coords는 현재 프레임에서 감지된 keypoint들의 score와 좌표를 나타내는 배열
# existing_poses는 이전 프레임의 포즈들을 나타내는 리스트
=======
# 입력으로 받은 poses와 squared_nms_radius, keypoint_scores, keypoint_coords 정보를 이용하여,
# 각 keypoint의 점수를 다른 포즈들과 비교하여 겹치지 않는 점수를 구하는 함수
>>>>>>> Stashed changes
def get_instance_score(
        existing_poses, squared_nms_radius,
        keypoint_scores, keypoint_coords):
    not_overlapped_scores = 0.
    for keypoint_id in range(len(keypoint_scores)):
        # 이전 프레임에서 찾은 포즈들과 겹친다면 제외, 겹치지 않을 시 더한다?
        if not within_nms_radius(
                existing_poses, squared_nms_radius,
                keypoint_coords[keypoint_id], keypoint_id):
            not_overlapped_scores += keypoint_scores[keypoint_id]

    # 반환 값은 현재 프레임에서 찾은 포즈의 score를 의미?
    return not_overlapped_scores / len(keypoint_scores)

<<<<<<< Updated upstream
# 위의 함수와 비슷한 기능을 수행 (입력으로 들어가는 배열이 다름)
=======
# get_instance_score 함수의 빠른 ver
>>>>>>> Stashed changes
def get_instance_score_fast(
        exist_pose_coords,
        squared_nms_radius,
        keypoint_scores, keypoint_coords):

    # 배열이 비어있는지 확인 후 비어있지 않다면 exist..배열의 모든 행과 keypoint... 배열의 모든 행 간의 거리 계산
    if exist_pose_coords.shape[0]:
        s = np.sum((exist_pose_coords - keypoint_coords) ** 2, axis=2) > squared_nms_radius
        not_overlapped_scores = np.sum(keypoint_scores[np.all(s, axis=0)])

    # 배열이 비어있다면 keypoint_scores 배열의 모든 값을 더하기    
    else:
        not_overlapped_scores = np.sum(keypoint_scores)

    # 반환값은 현재 프레임에서 찾은 포즈의 score를 의미 ?    
    return not_overlapped_scores / len(keypoint_scores)

<<<<<<< Updated upstream
# 특정 위치에서 해당 keypoint의 점수가 최대값인지 확인하는 함수 
=======
# 입력으로 받은 keypoint_id, score, hmy, hmx, local_max_radius, scores 정보를 이용하여,
# 주변 local_max_radius 내에서 현재 keypoint가 max score를 가지는 지 확인하는 함수
>>>>>>> Stashed changes
def score_is_max_in_local_window(keypoint_id, score, hmy, hmx, local_max_radius, scores):

    # scores의 배열의 높이와 너비를 height와 width로 나타내기
    height = scores.shape[0]
    width = scores.shape[1]
    
    # 상하좌우 좌표 계산 (최대, 최소 설정으로 경계값을 벗어나지 않도록 설정)
    y_start = max(hmy - local_max_radius, 0)
    y_end = min(hmy + local_max_radius + 1, height)
    x_start = max(hmx - local_max_radius, 0)
    x_end = min(hmx + local_max_radius + 1, width)

    # 각 키포인트의 점수 확인 (조건에 따라 True, false 반환)
    for y in range(y_start, y_end):
        for x in range(x_start, x_end):
            if scores[y, x, keypoint_id] > score:
                return False
    return True

<<<<<<< Updated upstream
# 주어진 정보를 이용하여 각각의 키포인트에 대한 score와 위치 정보를 구성하여 리스트로 반환하는 함수
=======
# 입력으로 받은 score_threshold, local_max_radius, scores 정보를 이용하여,
# keypoint의 정보와 좌표를 list로 반환하는 함수
>>>>>>> Stashed changes
def build_part_with_score(score_threshold, local_max_radius, scores):
    parts = []
    # scores의 배열의 높이와 너비, 키포인트 개수를 height와 width, num_keypoints로 나타내기
    height = scores.shape[0]
    width = scores.shape[1]
    num_keypoints = scores.shape[2]

    # 이미지에서 어떤 키포인트를 검출하거나 특징을 추출
    for hmy in range(height):
        for hmx in range(width):
            for keypoint_id in range(num_keypoints):
                score = scores[hmy, hmx, keypoint_id]
                if score < score_threshold:
                    continue
                if score_is_max_in_local_window(keypoint_id, score, hmy, hmx,
                                                local_max_radius, scores):
                    # 특정 위치에 해당하는 keypoint의 점수가 최대값일때만 리스트에 추가
                    parts.append((
                        score, keypoint_id, np.array((hmy, hmx))
                    ))
    return parts

<<<<<<< Updated upstream
import cv2

# 위의 함수와 비슷한 기능을 수행 ? (함수를 호출여부 차이인듯?)
=======
# build_part_with_score 함수의 빠른 ver
>>>>>>> Stashed changes
def build_part_with_score_fast(score_threshold, local_max_radius, scores):
    parts = []
    num_keypoints = scores.shape[2]
    lmd = 2 * local_max_radius + 1

    # 특정 위치에서 최대값을 가진 위치를 찾아서 parts 리스트에 추가 
    for keypoint_id in range(num_keypoints):
        kp_scores = scores[:, :, keypoint_id].copy()
        kp_scores[kp_scores < score_threshold] = 0  

        max_vals = ndi.maximum_filter(kp_scores, size=lmd, mode='constant')
        max_loc = np.logical_and(kp_scores == max_vals, kp_scores > 0)

        max_loc_idx = max_loc.nonzero()
        for y, x in zip(*max_loc_idx):
            parts.append((
                scores[y, x, keypoint_id],
                keypoint_id,
                np.array((y, x))
            ))
<<<<<<< Updated upstream
    #print(parts)
=======
    print(parts)
    # parts = 세개의 항목을 가진 튜플 
    # 첫번째 항목 = 키포인트 점수, 두번째 항목 = 키포인트의 ID?, 세번째 항목 = 키포인트의 위치 좌표
>>>>>>> Stashed changes

    return parts

<<<<<<< Updated upstream
# 여러 인물들의 포즈를 디코딩하는 함수
=======
# 여러 포즈를 추출해내는 함수
# 입력으로 받은 정보를 이용하여, 먼저 build_part_with_score_fast 함수를 이용하여
# keypoint 정보와 좌표를 얻은 후, 해당 정보를 이용하여 다수의 인간 포즈를 반환.
# 이 함수의 출력으로는 pose_scores, pose_keypoint_scores, pose_keypoint_coords 정보 반환
>>>>>>> Stashed changes
def decode_multiple_poses(
        scores, offsets, displacements_fwd, displacements_bwd, output_stride,
        max_pose_detections=10, score_threshold=0.5, nms_radius=20, min_pose_score=0.5):
        
        # score: 이미지 위의 각 위치에서 포즈 예측에 대한 값으로 이뤄진 3D numpy 배열
        #        배열의 세 번째 차원: 17개의 포인트 (목, 오른쪽 어깨, 왼쪽 어깨,
        #        오른쪽 팔꿈치, 왼쪽 팔꿈치, 오른쪽 손목, 왼쪽 손목,
        #        오른쪽 엉덩이, 왼쪽 엉덩이, 오른쪽 무릎, 왼쪽 무릎, 오른쪽 발목, 왼쪽 발목)
        #        에 대한 점수(해당 포인트가 이미지 상에서 실제로 존재할 확률)를 나타냄
        # offset: 2D numpy 배열, 이미지 상의 각 위치에서 예측된 이전 포즈와 현재 위치 간 차이
        # displacements_fwd: 4D numpy 배열, 현재 포인트에서 다음 포인트로 이동하는 경우의 이동량
        # displacements_bwd: 4D numpy 배열, 현제 포인트에서 이전 포인트로 이동하는 경우의 이동량
        # output_stride: ResNet의 네트워크 출력에 대한 스트라이드
        # max_pose_detections: 추출하려는 최대 포즈 수
        # score_threshold: 추출하려는 포즈에 대한 최소 점수
        # nms_radius: 포즈 겹치기 비율, 추출된 포즈들의 IOU가 해당 값 이상일 때 겹치는 것으로 판단
        # min_pose_score: 추출된 포즈의 최소 점수, 이 값보다 낮은 점수를 가진 포즈는 추출 X

    pose_count = 0
<<<<<<< Updated upstream
    # 각 인스턴스의 점수를 저장하는 배열
    pose_scores = np.zeros(max_pose_detections)
    # 각 인스턴스의 모든 키포인트의 점수를 저장하는 배열 (2차원)
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))
    # 각 인스턴스의 모든 키포인트의 좌표를 저장하는 배열 (3차원)
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))
=======
    pose_scores = np.zeros(max_pose_detections)  # 1×10 영행렬
    pose_keypoint_scores = np.zeros((max_pose_detections, NUM_KEYPOINTS))  # 10×각 부위 단어 길이 영행렬
    pose_keypoint_coords = np.zeros((max_pose_detections, NUM_KEYPOINTS, 2))  # 높이 10, 각 부위 단어 길이×2 영행렬
>>>>>>> Stashed changes

    squared_nms_radius = nms_radius ** 2

    # 함수를 통해 반환된값을 score_parts에 저장하고 점수가 높은 순서대로 정렬하기
    scored_parts = build_part_with_score_fast(score_threshold, LOCAL_MAXIMUM_RADIUS, scores)
    scored_parts = sorted(scored_parts, key=lambda x: x[0], reverse=True)
<<<<<<< Updated upstream
    # change dimensions from (h, w, x) to (h, w, x//2, 2) to allow return of complete coord array
=======

>>>>>>> Stashed changes
    height = scores.shape[0]
    width = scores.shape[1]
    # 각 배열을 재구성하고 마지막 두 축을 바꾸어 배열을 갱신
    offsets = offsets.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_fwd = displacements_fwd.reshape(height, width, 2, -1).swapaxes(2, 3)
    displacements_bwd = displacements_bwd.reshape(height, width, 2, -1).swapaxes(2, 3)
<<<<<<< Updated upstream
    print('init offsets : ', offsets)
    for root_score, root_id, root_coord in scored_parts:

=======

    
    for root_score, root_id, root_coord in scored_parts:

        # 각 값을 이용하여 root_point의 이미지 좌표를 계산
>>>>>>> Stashed changes
        root_image_coords = root_coord * output_stride + offsets[
            root_coord[0], root_coord[1], root_id]
      
        print('decode_multi_root_coord[0]: ', root_coord[0])
        print('decode_multi_root_coord[1]: ', root_coord[1])
        print('decode_multi_root_id : ',  root_id)
        print('decode_multi_offsets : ', offsets[root_coord[0], root_coord[1], root_id])


        # 현재 예측된 포즈와 중복되는지 확인
        if within_nms_radius_fast(
                pose_keypoint_coords[:pose_count, root_id, :], squared_nms_radius, root_image_coords):
            continue

        # 중복되지않는다면 decode_pose함수를 통해 root와 연결된 모든 부위의 포즈 정보를 계산    
        keypoint_scores, keypoint_coords = decode_pose(
            root_score, root_id, root_image_coords,
            scores, offsets, output_stride,
            displacements_fwd, displacements_bwd)

        # get_instance...함수를 통해 인스턴스에 대한 점수 계산
        pose_score = get_instance_score_fast(
            pose_keypoint_coords[:pose_count, :, :], squared_nms_radius, keypoint_scores, keypoint_coords)

<<<<<<< Updated upstream
        # 이 과정들을 반복하여 프레임에 있는 모든 포즈에 대한 정보를 담기

        # 감지된 포즈들에 대한 결과를 저장하는 과정
        # 조건 만족 여부에 따른 반복문 루프 종료 여부 결정
=======
        # NOTE this isn't in the original implementation, but it appears that by initially ordering by
        # part scores, and having a max # of detections, we can end up populating the returned poses with
        # lower scored poses than if we discard 'bad' ones and continue (higher pose scores can still come later).
        # Set min_pose_score to 0. to revert to original behaviour
        
>>>>>>> Stashed changes
        if min_pose_score == 0. or pose_score >= min_pose_score:
            pose_scores[pose_count] = pose_score
            pose_keypoint_scores[pose_count, :] = keypoint_scores
            pose_keypoint_coords[pose_count, :, :] = keypoint_coords
            pose_count += 1

        if pose_count >= max_pose_detections:
            break

    # 감지된 포즈의 개수, 포즈 점수, 포즈의 키포인트 좌표를 담고 있는 numpy 배열         
    return pose_scores, pose_keypoint_scores, pose_keypoint_coords
