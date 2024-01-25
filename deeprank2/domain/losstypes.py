from torch import nn

regression_losses = (
    nn.L1Loss,
    nn.SmoothL1Loss,
    nn.MSELoss,
    nn.HuberLoss,
)

binary_classification_losses = (
    nn.SoftMarginLoss,
    nn.BCELoss,
    nn.BCEWithLogitsLoss,
)

multi_classification_losses = (
    nn.CrossEntropyLoss,
    nn.NLLLoss,
    nn.PoissonNLLLoss,
    nn.GaussianNLLLoss,
    nn.KLDivLoss,
    nn.MultiLabelMarginLoss,
    nn.MultiLabelSoftMarginLoss,
)

other_losses = (
    nn.HingeEmbeddingLoss,
    nn.CosineEmbeddingLoss,
    nn.MarginRankingLoss,
    nn.TripletMarginLoss,
    nn.CTCLoss,
)

classification_losses = multi_classification_losses + binary_classification_losses

classification_tested = (
    nn.CrossEntropyLoss,
    nn.NLLLoss,
    nn.BCELoss,
    nn.BCEWithLogitsLoss,
)
