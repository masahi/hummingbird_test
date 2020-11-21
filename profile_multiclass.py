import time
import pickle

import numpy as np

import hummingbird.ml
from hummingbird.ml import constants


def benchmark(model, data, n_iter):
    for _ in range(3):
        model.predict(data)

    t1 = time.time()

    for _ in range(n_iter):
        # t11 = time.time()
        model.predict_proba(data)
        # t21 = time.time()
        # print(t21 - t11, data.shape)

    t2 = time.time()

    print((t2 - t1) / n_iter)


n_iter = 100

model = pickle.load(open("/home/masa/projects/dev/hummingbird/benchmarks/trees/models/covtype/xgb-500-6.pkl", "rb"))

device = "cuda"

batch_size = 10000

test_data = np.random.rand(batch_size, 54).astype(np.float32)
tvm_model = hummingbird.ml.convert(model, "tvm", test_data, device=device, extra_config={constants.BATCH_SIZE: 10000})

X = np.random.rand(batch_size, 54).astype(np.float32)

print("benchmarking baseline")
benchmark(model, X, n_iter)
print("benchmarking tvm")
benchmark(tvm_model, X, n_iter)
