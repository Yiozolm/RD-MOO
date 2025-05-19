from compressai.models import CompressionModel

__all__ = ['DemoCompressionModel']

class DemoCompressionModel(CompressionModel):
    def __init__(self, **kwargs):
        super(DemoCompressionModel, self).__init__(**kwargs)
