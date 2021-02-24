# Inspired by https://github.com/facebookresearch/hydra/blob/0.11.3/hydra/_internal/config_loader.py

import click
from pathlib import Path
from omegaconf import OmegaConf, DictConfig, ListConfig, open_dict
import copy
from pprint import pprint

def split_key_val(s):
    assert "=" in s, f"'{s}' not a valid override, expecting key=value format"

    idx = s.find("=")
    assert idx != -1
    return s[0:idx], s[idx + 1 :]

def get_yml_filepath(config_root, relative_conf_dir, config_name):
    ext = Path(config_name).suffix
    config_root = Path(config_root)

    yml_filename = config_name
    if ext not in (".yaml", ".yml"):
        yml_filepath= config_root / relative_conf_dir / f"{config_name}.yml"
        if yml_filepath.exists():
            return yml_filepath

        yml_filepath = config_root / relative_conf_dir / f"{config_name}.yaml"
        if yml_filepath.exists():
            return yml_filepath

        raise FileNotFoundError(
            f"No such file: {str(yml_filepath)}."
        )

    return config_root / relative_conf_dir / yml_filename

def load_defaults(config_root, defaults):
    defaults_config = OmegaConf.create()
    for default in defaults:
        assert isinstance(default, DictConfig), f"'defaults' entries should be pairs of key: value"
        # defaults entries are one-elements dicts
        relative_conf_dir, yml_filename = next(iter(default.items()))

        try:
            yml_filepath = get_yml_filepath(config_root, relative_conf_dir, yml_filename)
            # ignore_default=True to prevents defaults to be recursive
            current_default_config = load_yaml(yml_filepath, config_root, ignore_default=True)
            defaults_config = OmegaConf.merge(defaults_config, current_default_config)
        except FileNotFoundError as e:
            print(f'Tried to load default {relative_conf_dir}: {yml_filename} ...')
            print(e)

    return defaults_config

def apply_defaults_overrides(overrides, defaults):

    if overrides is None:
        return overrides

    key_to_idx = {}

    for idx, d in enumerate(defaults):
        if isinstance(d, DictConfig):
            key = next(iter(d.keys()))
            key_to_idx[key] = idx

    for override in copy.deepcopy(overrides):

        key, value = split_key_val(override)
        if key in key_to_idx:
            if value == "null":
                del defaults[key_to_idx[key]]
            else:
                defaults[key_to_idx[key]][key] = value

            overrides.remove(override)

        elif '.' not in key: # then it is a default entry wrongly named or a general option
            print(f"INFO: key {key} is not in defaults, so it is a general option.")


        # else it is not a default override, pass it to next function

    return overrides

def parse_defaults(config, config_root, overrides=None):
    defaults = config.get("defaults", None)

    if defaults is None:
        return config, overrides

    if not isinstance(defaults, ListConfig):
        raise AttributeError(
            f"Defaults must be a list, {type(defaults)} provided."
        )

    if overrides is not None:
        remaining_overrides = apply_defaults_overrides(overrides, defaults)
    else:
        remaining_overrides = None

    defaults_config = load_defaults(config_root, defaults)

    config.pop("defaults", None)

    return OmegaConf.merge(config, defaults_config), remaining_overrides


def parse_includes(config, yaml_filepath, config_root):
    includes = config.get("includes", None)

    if includes is None:
        return config

    if not isinstance(includes, ListConfig):
        raise AttributeError(
            f"Includes must be a list, {type(includes)} provided in {yaml_filepath}"
        )

    include_config = OmegaConf.create()


    for include in includes:
        include = Path(include)

        include_path = (config_root / include).absolute()

        # If path doesn't exist relative to config_root, try relative to current file
        if not include.exists():
            include_path = (yaml_filepath.parent / include).absolute()

        # recusive includes
        current_include_config = load_yaml(include_path, config_root, ignore_default=True)
        include_config = OmegaConf.merge(include_config, current_include_config)

    config.pop("includes", None)

    # config is coming from upper files in the recursion so has priority
    return OmegaConf.merge(include_config, config)

def load_yaml(yaml_filepath, config_root, ignore_default=False, overrides=None):

    if overrides == ['None']:
        overrides=None

    # Convert to absolute path for loading includes
    yaml_filepath = Path(yaml_filepath)

    config = OmegaConf.load(yaml_filepath.absolute())

    config = parse_includes(config, yaml_filepath, config_root)

    # option for defaults to not be recursive
    if not ignore_default:
        print(f"Parsing defaults of {yaml_filepath}")
        config, overrides = parse_defaults(config, config_root, overrides)
    else:
        config.pop("defaults", None)

    if overrides is not None and len(overrides) > 0:
        config.merge_with_dotlist(overrides)

    return config


@click.command()
@click.argument('config_root', type=click.Path(exists=True))
@click.argument('experiment_file', type=click.Path(exists=True, file_okay=True))
@click.argument('opts', nargs=-1, type=click.UNPROCESSED)
def main(config_root, experiment_file, opts):
    print(f"config_root: {config_root}")
    print(f"experiment_file: {experiment_file}")
    print(f"opts: {opts}")
    print('='*60)

    config_root = Path(config_root).absolute()

    config = load_yaml(experiment_file, config_root, overrides=list(opts))

    print(config.pretty())



if __name__ == "__main__":
    main()