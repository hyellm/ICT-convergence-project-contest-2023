import tensorflow as tf
import time
import argparse
import os

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--num_images', type=int, default=1000)
args = parser.parse_args()


def main():

    # TensorFlow 세션을 생성
    with tf.Session() as sess:
        # posenet.load_model 함수를 사용하여 모델을 로드
        # model_cfg = 모델의 구성정보 / model_outputs = 모델 출력값
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        # output_stride = 모델의 출력 스프라이드 값
        output_stride = model_cfg['output_stride']
        # num_images = 처리할 이미지 파일의 수 
        num_images = args.num_images

        # 확장자가 '.png' 또는 '.jpg'인 파일들을 골라내기
        filenames = [
            f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
        
        # 이미지 파일의 수가 해당조건을 만족할 경우 num_images개의 원소만 유지하도록 자르기
        if len(filenames) > num_images:
            filenames = filenames[:num_images]

        # 함수를 사용하여 이미지 파일 읽어 딕셔너리의 키 부분을 파일 경로로 저장 
        images = {f: posenet.read_imgfile(f, 1.0, output_stride)[0] for f in filenames}

        start = time.time()

        # 이미지들을 처리하는 반복문문
        for i in range(num_images):
            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                model_outputs,
                feed_dict={'image:0': images[filenames[i % len(filenames)]]}
            )

            # 이미지에서 추출된 사람의 포즈 정보를 디코딩하고 output 변수에 저장
            output = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        # 처리한 이미지의 수와 걸린 시간을 이용하여 평균 FPS 값을 계산 후 출력 
        print('Average FPS:', num_images / (time.time() - start))


if __name__ == "__main__":
    main()
