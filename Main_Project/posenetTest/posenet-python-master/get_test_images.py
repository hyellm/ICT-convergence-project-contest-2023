import urllib.request
import os
import argparse

GOOGLE_CLOUD_IMAGE_BUCKET = 'https://storage.googleapis.com/tfjs-models/assets/posenet/'

TEST_IMAGES = [
  'frisbee.jpg',
  'frisbee_2.jpg',
  'backpackman.jpg',
  'boy_doughnut.jpg',
  'soccer.png',
  'with_computer.jpg',
  'snowboard.jpg',
  'person_bench.jpg',
  'skiing.jpg',
  'fire_hydrant.jpg',
  'kyte.jpg',
  'looking_at_computer.jpg',
  'tennis.jpg',
  'tennis_standing.jpg',
  'truck.jpg',
  'on_bus.jpg',
  'tie_with_beer.jpg',
  'baseball.jpg',
  'multi_skiing.jpg',
  'riding_elephant.jpg',
  'skate_park_venice.jpg',
  'skate_park.jpg',
  'tennis_in_crowd.jpg',
  'two_on_bench.jpg',
]

parser = argparse.ArgumentParser()
parser.add_argument('--image_dir', type=str, default='./images')
args = parser.parse_args()

# TEST_IMAGES 리스트에 있는 이미지 파일들을 다운로드 받아 지정된 디렉토리에 저장
def main():
    # 디렉토리 존재 여부 확인 -> 존재하지 않는다면 디렉토리 생성
    if not os.path.exists(args.image_dir):
        os.makedirs(args.image_dir)

    # 리스트의 이미지 파일들을 하나씩 다운로드 받아 디렉토리에 저장 
    for f in TEST_IMAGES:
        url = os.path.join(GOOGLE_CLOUD_IMAGE_BUCKET, f)
        print('Downloading %s' % f)
        urllib.request.urlretrieve(url, os.path.join(args.image_dir, f))


if __name__ == "__main__":
    main()
