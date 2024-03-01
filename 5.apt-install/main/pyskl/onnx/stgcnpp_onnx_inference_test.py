import os
import numpy as np
import cv2
import argparse
import onnxruntime
from tqdm import tqdm
import time


def main():

    # modelPath = "stgcnpp_123_20230601_dynamic.onnx"
    modelPath = "stgcnpp_123_20230601_dynamic_simplify.onnx"

    session = onnxruntime.InferenceSession(modelPath, None,
                                           providers=['CUDAExecutionProvider']
                                            # providers=['TensorrtExecutionProvider']

                                          )
    input_name = session.get_inputs()[0].name
    eachkeypoints_seq = np.random.rand(1, 10, 2, 100, 17, 3).astype(np.float32)

    for i in range(1000):  
        t2 = time.time()
        outputs = session.run([], {input_name: eachkeypoints_seq})
        t1 = time.time()
        print(t1-t2)



if __name__== "__main__":
    main()