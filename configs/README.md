
use ??? value for None

e.g.:

in .yml
```
model: ???
```

when accessing config
```
>>> config.get('model', None) is None == True
```

## includes: 
- can import with path relative to given config dir or relative to current dir\
    e.g. in experiments/kitti/self_supervised.yml 
    ```
    includes:
        - losses/hinted_loss.yml
        - ../../losses/hinted_loss.yml
    ``` 
    are equivalent with abs_path_to/Depth/configs given as config dir
- imports are ordered (the last in the list is always the newer)
- imports can be recursive
    
## defaults
- defaults entries are always relative to the given config dir and are DictConfig (i.e.: pairs of *key: value*)\
    e.g.
    ```
    defaults:
        - losses: hinted_loss
        - model: monodepth18
    ```
    will loads the configs from ``path_to_config_dir/losses/hinted_loss.py`` and 
    ``path_to_config_dir/model/monodepth18.py``.
- defaults values can be overriden from the cli.\
    e.g., with the example from the last bullet point:
    ``python ./utils/config.py ./configs/ experiments/kitti/self_supervised.yml model=monodepth50``
