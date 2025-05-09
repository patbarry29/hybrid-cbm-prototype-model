from src.models import inception_v3


def ModelXtoC(pretrained, freeze, n_classes, n_concepts, use_aux=False, expand_dim=0):
    return inception_v3(
            pretrained=pretrained,
            freeze=freeze,
            n_classes=n_classes,
            aux_logits=use_aux,
            n_concepts=n_concepts,
            bottleneck=True,
            expand_dim=expand_dim
        )