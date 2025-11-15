from src.models.layers.detr.presnet import PResNet
from src.models.layers.detr.hybrid_encoder import HybridEncoder
from src.primitives.batch import IMAGE, PYRAMID, BatchDict
from src.primitives.model import BaseBackbone


class DETRBackbone(BaseBackbone):
    state_dict_location = ['ema', 'module']
    state_dict_replace = [
        ('backbone', 'backbone.presnet'),
        ('encoder.encoder', 'backbone.hybrid_yyyyy.yyyyy'),
        ('encoder', 'backbone.hybrid_yyyyy'),
        ('yyyyy', 'encoder')
    ]
    def __init__(
            self,
            presnet: PResNet,
            hybrid_encoder: HybridEncoder,
    ):
        super().__init__()
        self.presnet = presnet
        self.hybrid_encoder = hybrid_encoder

    def forward(self, batch: BatchDict) -> BatchDict:
        image = batch['image']['image'] / 255.0
        B, T, C, H, W = image.size()
        image = image.flatten(0, 1)

        feature = self.presnet(image)
        feature = self.hybrid_encoder(feature)
        batch['intermediate']['features_p'] = tuple([
            f.unflatten(0, (B, T))
            for f in feature
        ])

        return batch
