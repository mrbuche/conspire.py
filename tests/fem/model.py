import conspire.fem as fem


def test_model_is_public_and_block_is_not():
    assert hasattr(fem, "Model")
    assert not hasattr(fem, "Block")
