import inspect

from .fno import TFNO, TFNO1d, TFNO2d, TFNO3d
from .fno import FNO, FNO1d, FNO2d, FNO3d
from .uno import UNO


MODEL_ZOO = {
    "tfno": TFNO,
    "tfno1d": TFNO1d,
    "tfno2d": TFNO2d,
    "tfno3d": TFNO3d,
    "fno": FNO,
    "fno1d": FNO1d,
    "fno2d": FNO2d,
    "fno3d": FNO3d,
    "uno": UNO,
}


def available_models():
    """List the available neural operators"""
    return list(MODEL_ZOO.keys())


def get_model(config):
    """Returns an instantiated model for the given config

    Also prints warnings for safety, in case::
    * some given arguments aren't actually used by the model
    * some keyword arguments of the models aren't provided by the config

    Parameters
    ----------
    config : Bunch or dict-like
        configuration, must have
        arch = config['arch'] (string)
        and the corresponding config[arch] (a subdict with the kwargs of the model)

    Returns
    -------
    model : nn.Module
        the instanciated module
    """
    arch = config["arch"].lower()
    config_arch = config.get(arch)

    # Set the number of input channels depending on channels in data + mg patching
    data_channels = config_arch.pop("data_channels")
    try:
        patching_levels = config["patching"]["levels"]
    except KeyError:
        patching_levels = 0
    if patching_levels:
        data_channels *= patching_levels + 1
    config_arch["in_channels"] = data_channels

    # Dispatch model creation
    try:
        return dispatch_model(MODEL_ZOO[arch], config_arch)
    except KeyError:
        raise ValueError(f"Got config.{arch=}, expected one of {MODEL_ZOO.keys}.")


def dispatch_model(ModelClass, config):
    """This function just creates an instance of the model ModelClass(**config)
    but performs additional checks to warn users about missing/wrong arguments.

    Parameters
    ----------
    ModelClass : nn.Module
        model to instancite
    config : Bunch or dict-like

    Returns
    -------
    ModelClass(**config) : instanciated model
    """
    sig = inspect.signature(ModelClass)
    model_name = ModelClass.__name__

    # Verify that given parameters are actually arguments of the model
    for key in config:
        if key not in sig.parameters:
            print(f"Given argument {key=} that is not in {model_name}'s signature.")
            # warnings.warn(f"Given argument {key=} that is not in {model_name}'s signature.")

    # Check for model arguments not specified in the configuration
    for key, value in sig.parameters.items():
        if (value.default is not inspect._empty) and (key not in config):
            print(
                f"Keyword argument {key} not specified for model {model_name}, "
                f"using {value.default=}."
            )
            # warnings.warn(
            #     f"Keyword argument {key} not specified for model {model_name}, "
            #     f"using default={value.default}.")

    return ModelClass(**config)
