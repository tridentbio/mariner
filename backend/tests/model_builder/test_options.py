from fleet.model_builder import options


# Tests if model_options uses information from file at model_builder/component_overrides.yml
# to get default_args and args_options
def test_get_model_options():
    model_options = options.get_model_options()
    tfe_option = None
    emb_option = None
    for option in model_options:
        if option.class_path == "torch.nn.TransformerEncoderLayer":
            tfe_option = option
        elif option.class_path == "torch.nn.Embedding":
            emb_option = option
    assert tfe_option and emb_option, "Options are missing from list"
    assert emb_option.default_args
    assert (
        "max_norm" in emb_option.default_args
        and "norm_type" in emb_option.default_args
    ), "Default args are missing"
    # checks if emb_option has defaults max_norm = 1 and norm_type = 2
    assert (
        emb_option.default_args["max_norm"] == 1
        and emb_option.default_args["norm_type"] == 2
    ), "Default args are not correct"
    assert tfe_option.args_options
    # checks if tfe_option has args_options with activation
    assert "activation" in tfe_option.args_options, "Args options are missing"
    # checks if tfe option args_options activation has [ 'relu', 'sigmoid' ]
    assert tfe_option.args_options["activation"] == [
        "relu",
        "sigmoid",
    ], "Args options are not correct"
    # checks if tfe_option.defaults_args is a dictionary and
    # contains ('activation', 'relu')
    assert isinstance(tfe_option.default_args, dict)
    assert "activation" in tfe_option.default_args
    assert tfe_option.default_args["activation"] == "relu"
