import torch
from utils.misc import make_list, same_shape
from collections import OrderedDict
from utils.colored_terminal_logging_utils import get_terminal_logger

terminal_logger = get_terminal_logger(__name__)

def load_tri_network(network, path, prefixes='model'):
    """
    Loads a pretrained network

    Weights from PackNet's repo are in the following format:
    >>> import torch
    >>> path = "./ResNet18_MR_selfsup_K.ckpt"
    >>> ckpt = torch.load(path, map_location='cpu')
    >>> list(ckpt.keys())
    ['config', 'state_dict']
    >>> list(ckpt['state_dict'].keys())
    ['model.depth_net.encoder.encoder.conv1.weight', 'model.depth_net.encoder.encoder.bn1.weight', ...,
     'model.pose_net.decoder.net.3.weight', 'model.pose_net.decoder.net.3.bias']

     Hence, we discard the 'model.' prefix ion each of the params

    Note: yacs package is necessary to load Packnet's weights

    Parameters
    ----------
    network : nn.Module
        Network that will receive the pretrained weights
    path : str
        File containing a 'state_dict' key with pretrained network weights
    prefixes : str or list of str
        Layer name prefixes to consider when loading the network
    Returns
    -------
    network : nn.Module
        Updated network with pretrained weights
    """
    prefixes = make_list(prefixes)

    saved_state_dict = torch.load(path, map_location='cpu')['state_dict']

    # Get network state dict
    network_state_dict = network.state_dict()

    updated_state_dict = OrderedDict()
    n, n_total = 0, len(network_state_dict.keys())
    for key, val in saved_state_dict.items():
        for prefix in prefixes:
            prefix = prefix + '.'
            if prefix in key:
                idx = key.find(prefix) + len(prefix)
                key = key[idx:]
                if key in network_state_dict.keys() and \
                        same_shape(val.shape, network_state_dict[key].shape):
                    updated_state_dict[key] = val
                    n += 1

    network.load_state_dict(updated_state_dict, strict=False)
    terminal_logger.info(f"Loaded {n}/{n_total} tensors from {path}")