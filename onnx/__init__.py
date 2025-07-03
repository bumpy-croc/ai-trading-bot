# Minimal stub for the `onnx` package – only to satisfy imports in tests.
class ModelProto:
    pass

def load_model(*args, **kwargs):
    return ModelProto()

# Expose submodules often imported
import types, sys as _sys
_helper = types.ModuleType('onnx.helper')
_helper.make_tensor = lambda *a, **kw: None
_sys.modules['onnx.helper'] = _helper