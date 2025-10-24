import os
import tensorflow as tf
import sys

print("=" * 50)
print("Python 执行路径:", sys.executable)
print("CUDA_PATH 环境变量:", os.environ.get('CUDA_PATH', '未设置'))
print("PATH 环境变量:", os.environ.get('PATH', '未设置'))
print("=" * 50)

# 检查 TensorFlow 是否找到 GPU
print(f"TensorFlow 版本: {tf.__version__}")
print(f"GPU 设备列表: {tf.config.experimental.list_physical_devices('GPU')}")
print(f"GPU 可用性: {tf.test.is_gpu_available()}")
print(f"构建时是否支持 CUDA: {tf.test.is_built_with_cuda()}")

# 尝试一个简单的 GPU 操作
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
    print("GPU 计算测试成功!")
    print(c)
except Exception as e:
    print("GPU 计算测试失败:", e)