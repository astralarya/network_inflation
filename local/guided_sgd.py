from torch.optim import SGD


class GuidedSGD(SGD):
    def __init__(
        self,
        params,
        guide,
        guide_alpha=1.,
        guide_epochs=32,
        **kwargs,
    ) -> None:
        super().__init__(params, **kwargs)
        self.guide = guide
        self.guide_alpha = guide_alpha
        self.guide_epochs = guide_epochs
