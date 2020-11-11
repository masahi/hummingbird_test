import onnx
import tvm
from tvm import relay
from tvm.runtime.vm import VirtualMachine


onnx_model = onnx.load("data/hb_fraud.onnx")

ishape = {"input_0": [relay.Any(), 28]}

model, params = relay.frontend.from_onnx(onnx_model, ishape)

print(model)

target = "llvm"
config = {"relay.FuseOps.max_depth": 50}

with tvm.transform.PassContext(opt_level=3, disabled_pass=["FoldScaleAxis"], config=config):
    vm_exec = relay.vm.compile(model, target=target, params=params)
