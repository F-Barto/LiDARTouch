Project configuration specifies the environment setups, resources paths and security,
which mostly are from engineering aspects. Separating this configuration from the models' hyperparameters
enables us to automatically search for the best hyperparameters using other ML libraries.

each project config is in the format:

```
output_dir: /home/clear/fbartocc/output_data/MultiCamDepth
project_name: MultiCamMonoDepthArgo # will also be used as wandb project name
experiment_name: self_supervised_packnet # will also be used as wandb experiment name, each run have a unique id
```

the output data (weights, imgs, ...) will be stored at {output_dir}/{project_name}/{experiment_name}-{run_id}
a project have multiple experiment (different learning scheme, architecture, etc...)
an experiment will have different hyperparameters (learning rate, layers width, depth of arch etc...)
each experiment run will have a directory named with a unique id under the experiment folder.
experiment_name: self_supervised_packnet -> self_supervised_packnet/2ej5d4, self_supervised_packnet/5e742g