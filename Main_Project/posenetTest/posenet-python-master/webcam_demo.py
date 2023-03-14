#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import time
import argparse
import socket
import numpy as np

import posenet
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7) #0.7125
parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
args = parser.parse_args()


def main():
    # TensorFlow 세션을 시작하고 모델과 값을 가져오기
    with tf.Session() as sess:
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # 카메라 캡쳐
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        # start = 시작 시간 변수 -> 초기화 
        start = time.time()
        # frame_count = 프레임 수를 저장할 변수 -> 초기화 
        frame_count = 0


        while True:

            # 웹캠에서 프레임을 읽어들임
            input_image, display_image, output_scale = posenet.read_cap(
                cap, scale_factor=args.scale_factor, output_stride=output_stride)

            # sess.run 함수 = 입력 이미지를 모델에 전달하고 예측 결과를 반환
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            ) 
            # feed_dict = 입력 이미지 전달
            # 반환된 예측 결과
            # heatmaps_result = 각각의 관절에 대한 정보?
            # offsets_result = 각 관절에서 다른 관절까지의 오프셋
            # displacement_fwd_result = 앞쪽 방향으로의 변위
            # displacement_bwd_result = 뒷쪽 방향으로의 범위 


            # posenet.decode_...함수를 사용하여 해독 -> 예측된 포즈 점수, 키포인트 점수, 좌표를 제공
            # decode_multiple_poses = 여러개의 사람 포즈를 추정하고, 추정된 포즈들의 점수와 좌표를 반환
            # sqeeze 함수 = 4차원 배열에서 3차원 배열로 변환하는 함수
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15)

            # 포즈 좌표를 실제 이미지 좌표로 변환
            keypoint_coords *= output_scale

            # 변환된 좌표를 이용하여 draw_skel_and_kp 함수를 이용하여 원본 이미지에 포즈를 시각화
            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            # 이미지 시각화를 위해 포즈를 시각화한 이미지를 보여주기
            cv2.imshow('posenet', overlay_image)
            frame_count += 1

            # 'q'를 누르면 프로그램 종료 
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 처리 속도를 측정하여 평균 FPS 값을 출력        
        print('Average FPS: ', frame_count / (time.time() - start))

if __name__ == "__main__":
    main()