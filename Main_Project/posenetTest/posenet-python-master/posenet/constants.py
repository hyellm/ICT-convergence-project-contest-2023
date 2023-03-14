# 사람의 자세를 추정하는데 사용되는 정보들을 제공

# PART_NAMES = 인체의 주요 부위(17개)의 이름을 담은 리스트 
PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

# NUM_KEYPOINTS = 부위의 개수를 나타내는 변수 
NUM_KEYPOINTS = len(PART_NAMES)

# PART_IDS = 부위 이름을 해당 부위의 인덱스로 매핑한 딕셔너리
PART_IDS = {pn: pid for pid, pn in enumerate(PART_NAMES)}

# CONNECTED_PART_NAMES = 연결된 부위들의 쌍으로 나타내는 리스트 
CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

# CONNECTED_PART_INDICES = 연결된 부위들의 인덱스 쌍을 나타내는 리스트
CONNECTED_PART_INDICES = [(PART_IDS[a], PART_IDS[b]) for a, b in CONNECTED_PART_NAMES]

# LOCAL_MAXIMUM_RADIUS = Keypoint 검출시 연관된 인근 영역의 크기
LOCAL_MAXIMUM_RADIUS = 1

# POSE_CHAIN = 인체 포즈 추정시 이전 부위와 연결된 부위를 나타내는 리스트
POSE_CHAIN = [
    ("nose", "leftEye"), ("leftEye", "leftEar"), ("nose", "rightEye"),
    ("rightEye", "rightEar"), ("nose", "leftShoulder"),
    ("leftShoulder", "leftElbow"), ("leftElbow", "leftWrist"),
    ("leftShoulder", "leftHip"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("nose", "rightShoulder"),
    ("rightShoulder", "rightElbow"), ("rightElbow", "rightWrist"),
    ("rightShoulder", "rightHip"), ("rightHip", "rightKnee"),
    ("rightKnee", "rightAnkle")
]

# PARENT_CHILD_TUPLES = POSE_CHAIN 리스트를 부위 인덱스 쌍으로 나타내는 리스트
PARENT_CHILD_TUPLES = [(PART_IDS[parent], PART_IDS[child]) for parent, child in POSE_CHAIN]

# PART_CHANNELS = 출력 마스크에서 각 부위의 색상 채널 이름을 나타내는 리스트 
PART_CHANNELS = [
  'left_face',
  'right_face',
  'right_upper_leg_front',
  'right_lower_leg_back',
  'right_upper_leg_back',
  'left_lower_leg_front',
  'left_upper_leg_front',
  'left_upper_leg_back',
  'left_lower_leg_back',
  'right_feet',
  'right_lower_leg_front',
  'left_feet',
  'torso_front',
  'torso_back',
  'right_upper_arm_front',
  'right_upper_arm_back',
  'right_lower_arm_back',
  'left_lower_arm_front',
  'left_upper_arm_front',
  'left_upper_arm_back',
  'left_lower_arm_back',
  'right_hand',
  'right_lower_arm_front',
  'left_hand'
]