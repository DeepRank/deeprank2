from deeprank2.utils.earlystopping import EarlyStopping

dummy_val_losses = [3, 2, 1, 2, 0.5, 2, 3, 4, 5, 6, 7]
dummy_train_losses = [3, 2, 1, 2, 0.5, 2, 3, 4, 5, 1, 7]


def base_earlystopper(patience=10, delta=0, maxgap=None):
    early_stopping = EarlyStopping(
        patience=patience,
        delta=delta,
        maxgap=maxgap,
        min_epoch=0,
    )

    for ep, loss in enumerate(dummy_val_losses):
        # check early stopping criteria
        print(f"Epoch #{ep}", end=": ")
        early_stopping(ep, loss, dummy_train_losses[ep])
        if early_stopping.early_stop:
            break

    return ep


def test_patience():
    patience = 3
    final_ep = base_earlystopper(patience=patience)
    # should terminate at epoch 7
    assert final_ep == 7


def test_patience_with_delta():
    patience = 3
    delta = 1
    final_ep = base_earlystopper(patience=patience, delta=delta)
    # should terminate at epoch 5
    assert final_ep == 5


def test_maxgap():
    maxgap = 1
    final_ep = base_earlystopper(maxgap=maxgap)
    # should terminate at epoch 9
    assert final_ep == 9
