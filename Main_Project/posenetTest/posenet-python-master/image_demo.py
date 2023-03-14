import tensorflow as tf
import cv2
import time
import argparse
import os

import posenet


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

# TensorFlow를 사용하여 이미지에서 자세 감지를 수행하는 코드 
def main():

    # TensorFlow를 사용하여 모델을 로드하는 코드
    with tf.Session() as sess:
        # posenet.load_model 함수를 사용하여 모델을 로드
        # model_cfg = 모델의 구성정보 / model_outputs = 모델 출력값
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
         # output_stride = 모델의 출력 스프라이드 값
        output_stride = model_cfg['output_stride']

        # 출력 디렉토리가 지정되어 있는 경우 디렉토리가 존재하지 않으면 디렉토리 생성 
        if args.output_dir:
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

        # 확장자가 '.png' 또는 '.jpg'인 파일들을 골라내기
        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

        start = time.time()

        # 이미지들을 하나씩 불러와서 모델에 입력으로 넣고, 모델의 출력 결과를 이용하여
        # 각 관절의 위치, 각 이미지에서 찾아진 사람의 포즈 점수를 계산 

        # 출력 결과 = heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result
        # 각 관절의 위치 = keypoint_coords / 각 이미지에서 찾아진 사람의 포즈 점수 = pose_scores
        for f in filenames:
            input_image, draw_image, output_scale = posenet.read_imgfile(
                f, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': input_image}
            )

            # 사람의 포즈를 추정하는 부분 = 이를 3개의 변수에 저장
            # pose_scores = 각 포즈에 대한 점수 목록
            # keypoint_scores = 모든 포즈에서 모든 키포인트가 포함된 3차원 TensorFlow
            # keypoint_coords = 모든 포즈에서 모든 키포인트의 좌표가 포함된 3차원 TensorFlow
            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

            # 이미지의 크기에 맞게 관절 위치 조정 
            keypoint_coords *= output_scale

            # 디렉토리가 지정되어 있다면, 추출한 포즈에 대해 스켈레톤 작업 및 키포인트 정보를 이미지에 그린다음
            # 그린 이미지를 디렉토리에 저장
            if args.output_dir:
                draw_image = posenet.draw_skel_and_kp(
                    draw_image, pose_scores, keypoint_scores, keypoint_coords,
                    min_pose_score=0.25, min_part_score=0.25)

                cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

            # notxt옵션? 이 켜져 있지 않을 경우 ->  포즈와 해당 키포인트 좌표 출력  
            if not args.notxt:
                print()
                print("Results for image: %s" % f)
                for pi in range(len(pose_scores)):
                    if pose_scores[pi] == 0.:
                        break
                    print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                    for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                        print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        # 처리된 이미지 수에 따른 처리 속도를 계산하여 출력
        print('Average FPS:', len(filenames) / (time.time() - start))


if __name__ == "__main__":
    main()
