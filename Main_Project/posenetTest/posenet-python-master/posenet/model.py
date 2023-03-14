#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
import posenet.converter.config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODEL_DIR = './_models'
DEBUG_OUTPUT = False

# model_id에 따라 ordinal 값을 반환하는 함수 
# ordinal = 순서에 따라 지정된 값
def model_id_to_ord(model_id):

    # 각 조건에 맞는 ordinal의 값 반환 
    if 0 <= model_id < 4:
        return model_id  
    elif model_id == 50:
        return 0
    elif model_id == 75:
        return 1
    elif model_id == 100:
        return 2
    else:  # 101
        return 3


# model_ord를 입력으로 받아 posenet을 로드하기 위한 설정 정보를 반환하는 함수
def load_config(model_ord):
    # posenet.converter...함수를 사용하여 설정 파일을 불러오기
    # checkpoints, output_stride 값 가져오기 
    converter_cfg = posenet.converter.config.load_config()
    checkpoints = converter_cfg['checkpoints']
    output_stride = converter_cfg['outputStride']
    checkpoint_name = checkpoints[model_ord]

    # output_stride와 checkpoint_name을 딕셔너리 형태로 만들어 반환
    model_cfg = {
        'output_stride': output_stride,
        'checkpoint_name': checkpoint_name,
    }

    return model_cfg


# model_id를 이용하여 posenet 파일을 불러오는 함수 
def load_model(model_id, sess, model_dir=MODEL_DIR):
    
    # model_id 값을 함수에 활용하여 model_ord 값으로 변환
    model_ord = model_id_to_ord(model_id)
    # model_ord 값을 함수에 활용하여 설정 정보로 변환
    model_cfg = load_config(model_ord)
    # 파일 경로 생성 ?
    model_path = os.path.join(model_dir, 'model-%s.pb' % model_cfg['checkpoint_name'])

    # 경로에 파일이 있는지 확인
    # 파일이 없다면 Tensorflow Python 모델로 변환하여 저장 
    if not os.path.exists(model_path):
        print('Cannot find model file %s, converting from tfjs...' % model_path)
        from posenet.converter.tfjs2python import convert
        convert(model_ord, model_dir, check=False)
        assert os.path.exists(model_path)

    with tf.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # 동작 X
    if DEBUG_OUTPUT:
        graph_nodes = [n for n in graph_def.node]
        names = []
        for t in graph_nodes:
            names.append(t.name)
            print('Loaded graph node:', t.name)

    offsets = sess.graph.get_tensor_by_name('offset_2:0')
    displacement_fwd = sess.graph.get_tensor_by_name('displacement_fwd_2:0')
    displacement_bwd = sess.graph.get_tensor_by_name('displacement_bwd_2:0')
    heatmaps = sess.graph.get_tensor_by_name('heatmap:0')

    return model_cfg, [heatmaps, offsets, displacement_fwd, displacement_bwd]
