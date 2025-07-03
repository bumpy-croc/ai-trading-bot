# Lightweight stub for `onnxruntime` used in strategy modules.
class InferenceSession:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return []