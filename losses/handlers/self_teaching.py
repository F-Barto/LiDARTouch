


def self_teaching_loss_handler(batch, preds, poses, progress, monoscale_loss_fn, selfteaching_loss):

    losses =[]
    metrics = {}

    cam_hinted_output = monoscale_loss_fn(
        batch['target_view_original'],
        batch['source_views_original'],
        preds['cam_disp'],
        batch['sparse_projected_lidar_original'],
        batch['intrinsics'],
        poses,
        progress=progress)
    cam_losses = cam_hinted_output['loss']
    cam_metrics = {'cam/' + k: v for k, v in cam_hinted_output['metrics'].items()}

    lidar_hinted_output = monoscale_loss_fn(
        batch['target_view_original'],
        batch['source_views_original'],
        preds['lidar_disp'],
        batch['sparse_projected_lidar_original'],
        batch['intrinsics'],
        poses,
        progress=progress)
    lidar_losses = lidar_hinted_output['loss']
    lidar_metrics = {'lidar/' + k: v for k, v in lidar_hinted_output['metrics'].items()}

    # cam_losses, cam_metrics = self.compute_common_losses_and_metrics(batch, preds['cam_disp'], poses, progress, metrics_prefix='cam/')
    # lidar_losses, lidar_metrics = self.compute_common_losses_and_metrics(batch, preds['lidar_disp'], poses, progress, metrics_prefix='lidar/')

    losses += cam_losses + lidar_losses
    metrics.update({**cam_metrics, **lidar_metrics})

    if 'uncertainties' in preds:
        weights = preds.get('adaptive_weights', None)

        selfteaching_output = selfteaching_loss([preds['cam_disp'][0], preds['lidar_disp'][0]],
                                                      preds['inv_depths'], preds['uncertainties'],
                                                      weights=weights)
        losses.append(selfteaching_output['loss'])
        metrics.update(selfteaching_output['metrics'])

    return losses, metrics