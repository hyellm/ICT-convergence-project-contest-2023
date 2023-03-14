import cv2 
import numpy as np  

import posenet.constants


# 유효한 해상도를 계산하는 함수
# 입력 : 이미지의 너비(width), 높이(height), 출력 스트라이드(output_stride)
def valid_resolution(width, height, output_stride=16):

    # 유효한 해상도 계산
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1

    # 반환값은 (target_width, target_height) 형태의 튜플로 반환
    return target_width, target_height


# 입력 이미지를 처리하는 함수 
# 입력 이미지는 source_img로 제공, scale_factor와 output_stride 매개변수를 사용하여 이미지를 전처리
def _process_input(source_img, scale_factor=1.0, output_stride=16):

    # valid_resolution 함수를 통해 유효한 해상도를 계산
    target_width, target_height = valid_resolution(
        source_img.shape[1] * scale_factor, source_img.shape[0] * scale_factor, output_stride=output_stride)
    
    # scale = 이미지를 처리하기 위해 사용되는 크기 비율
    # 축소된 비율을 이용해 배열 만들고 이를 이용해 후속 처리 단계에서 원하는 크기로 확장 
    scale = np.array([source_img.shape[0] / target_height, source_img.shape[1] / target_width])
    #print('target_width = ', target_width)
    #print('target_height = ', target_height)
    
    # cv2의 resize함수를 이용하여 source_img를 target_width와 target_height에 맞게 조정
    # resize 함수 = 이미지의 크기를 조절하는 함수 
    # cv2.INTER_LINEAR 함수 = 양선형 보간법 (효율성이 가장 좋음, 속도 빠름, 퀄리티 적당)
    input_img = cv2.resize(source_img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    
    # cv2의 cvtColor함수를 이용하여 BGR 색상 공간에서 RGB 색상 공간으로 변환
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).astype(np.float32)
    
    # 이미지를 픽셀 단위로 정규화 (픽셀 범위 : 0 ~ 255 -> -1 ~ 1)
    input_img = input_img * (2.0 / 255.0) - 1.0

    # reshape함수를 이용하여 4차원을 가지는 배열로 변환
    # 첫번째 차원 : 배치 크기 (하나의 이미지만 처리하므로 1) / 두번째, 세번째 차원 : 이미지의 높이와 너비 / 네번째 차원 : 색상 채널
    input_img = input_img.reshape(1, target_height, target_width, 3)
    
    # 반환값은 전처리된 입력 이미지(input_img), 원본 입력 이미지(source_img), 크기 조정시 사용된 크기 비율(scale)
    return input_img, source_img, scale


# 웹캠에서 읽은 영상 프레임을 입력으로 받아 함수 호출을 통한 처리 후 반환하는 함수
def read_cap(cap, scale_factor=1.0, output_stride=16):

    # cap.read 함수를 이용하여 프레임을 읽어오기
    # 비디오 프레임을 제대로 읽으면 res가 true, 실패시 false  / 읽은 프레임은 img
    res, img = cap.read()

    # res가 false인 경우 
    if not res:
        # 예외 발생 -> 함수 종료
        raise IOError("webcam failure")

    # 반환값은 읽은 이미지를 함수를 통해 처리한 값    
    return _process_input(img, scale_factor, output_stride)


# 이미지 파일 경로를 입력으로 받아 해당 경로의 이미지를 함수 호출을 통한 처리 후 반환하는 함수 
def read_imgfile(path, scale_factor=1.0, output_stride=16):
    
    # 파일 경로에서 읽어온 이미지를 img에 저장
    img = cv2.imread(path)

    # 반환값은 읽은 이미지를 함수를 통해 처리한 값
    return _process_input(img, scale_factor, output_stride)

# 이미지에 키포인트를 그리는 함수 
# 입력 : 입력 이미지 / 인스턴스 점수 / 각 키포인트의 점수(2D) / 각 키포인트의 좌표(3D) 
#        / 그리지 않을 최소 인스턴스 점수 / 그리지 않을 최소 키포인트 점수
def draw_keypoints(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):

    cv_keypoints = []

    # 이중 for문을 통해 각 조건을 만족하는 경우 cv2.KeyPoint를 생성하고 cv_keypoints리스트에 추가
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue

        # 해당 인스턴스의 모든 키포인트 점수와 좌표 가져오기 (해당 인덱스 값 하나씩 가져옴)
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_confidence:
                continue
            # cv2.KeyPoint를 생성하고 cv_keypoints리스트에 추가
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

    # 이미지 위에 키포인트 그리기         
    out_img = cv2.drawKeypoints(img, cv_keypoints, outImage=np.array([]))
    
    # 반환값은 주어진 색상과 크기로 키포인트를 그린 이미지 
    return out_img


# 점의 좌표를 가지고 오는 함수 (연결된 키포인트들의 좌표를 계산)
def get_adjacent_keypoints(keypoint_scores, keypoint_coords, min_confidence=0.1):
    results = [] #리턴시킬 배열 (x, y)
    
    # posenet에서 받아온 인덱스들의 좌 우를 찍음
    # posenet.CONNECTED_PART_INDICES는 연결된 키포인트 쌍들을 나타내는 상수
    for left, right in posenet.CONNECTED_PART_INDICES:
        if keypoint_scores[left] < min_confidence or keypoint_scores[right] < min_confidence:
            continue
        
        #받아온 결과를 numpy.array 형식으로 [y, x] 와 같이 저장 (::의 역할 / opencv에서는 좌표값을 y,x 순서로 다룸)
        results.append(
            np.array([keypoint_coords[left][::-1], keypoint_coords[right][::-1]]).astype(np.int32),
        )
   
    # 반환값은 좌표값 
    return results

# 이미지 위에 스켈레톤(뼈대)을 그리는 함수 
def draw_skeleton(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_confidence=0.5, min_part_confidence=0.5):
    out_img = img
    adjacent_keypoints = []
    
    for ii, score in enumerate(instance_scores):
        if score < min_pose_confidence:
            continue

        # get_adjacent_keypoints 함수에서 얻은 좌표들을 new_keypoints에 저장 (np.array 형태)
        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_confidence)
        
        # new_keypoints에 있는 좌표를 adjacent_keypoints 리스트에 추가
        adjacent_keypoints.extend(new_keypoints)

    # cv2.polylines 함수를 이용하여 스켈레톤 그리기 (이미지, 다각형 좌표, 다각형 닫힘 여부, 색상 전달)    
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))
    
    # 반환값은 스켈레톤이 그려진 이미지 반환
    return out_img

# 입력 이미지 위에 PoseNet 알고리즘이 예측한 결과를 시각화하여 출력 이미지를 반환하는 함수
# 위의 함수와 차이점은 좌표 표시 유무로 보임
def draw_skel_and_kp(
        img, instance_scores, keypoint_scores, keypoint_coords,
        min_pose_score=0.5, min_part_score=0.5):
    out_img = img
    adjacent_keypoints = []
    cv_keypoints = []
    component = []

    coord = []

    # 01. get_adjacent...함수를 사용하여 이웃 키포인트 쌍의 목록을 가져오기
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        
        adjacent_keypoints.extend(new_keypoints) # --- 여기까지가 위의 함수와 동일한 부분

        # 02. drawKeypoints 및 polylines를 사용하여 이웃하는 키포인트끼리 연결하는 뼈대 그리고 좌표 표시하기
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue

            # 조건을 만족한 경우에만 component 리스트에 추가
            x = kc[1].astype(np.int32)
            y = kc[0].astype(np.int32)
            component.append(x)
            component.append(y)

            # cv2.KeyPoint 객체를 생성하여 cv_keypoints 리스트에 추가
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 10. * ks))

        # component 리스트를 coord 리스트에 추가
        coord.extend(component)

    temp = []
    # 글꼴 설정
    font=cv2.FONT_HERSHEY_SIMPLEX
    i = 0

    while(True):
        # coord 길이가 0일 경우 반복문 중단
        if len(coord) == 0:
            break

        # temp 리스트에 coord의 i번째와 i+1번째 항목을 추가
        temp.append(coord[i])
        temp.append(coord[i+1])
        
        # text 변수 = (첫번째항목, 두번째 항목) 형식의 문자열 저장
        text = '({}, {})'.format(temp[0], temp[1])

        # out_img에 text 문자열을 temp 위치에 지정된 폰트로 출력
        out_img = cv2.putText(out_img, text, temp, font, 1, (255,0,0), 1)
        # temp 리스트 비우기
        temp.clear()
        i += 2

        # i의 값이 coord의 길이보다 클 경우 반복문 중단
        if i >= len(coord):
            break

    # out_img에 cv_keypoints 변수에 저장된 키포인트를 이용하여 키포인트 그리기 (색상도 지정, 플래그 사용)  
    # cv2.DRAW... = 키포인트에 있는 사이즈나 앵글에 들어있는 변수를 고려하여 다양한 크기와 직선을 이용해서 표현
    out_img = cv2.drawKeypoints(
        out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
      
    # cv2.polylines 함수를 이용하여 스켈레톤 그리기 (이미지, 다각형 좌표, 다각형 닫힘 여부, 색상 전달)  
    out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0))

    # 반환값은 스켈레톤이 그려진 이미지 반환
    return out_img
