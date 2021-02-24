SCRIPTS_DIR="../../data_preparation/argoverse"
DATA_ROOT_DIR="/home/clear/fbartocc/data/ARGOVERSE/argoverse-tracking"
OUTPUT_BASE_DIR="/home/clear/fbartocc/working_data"

# camera-lidar sync and calibrations data collection
python $SCRIPTS_DIR/synchronization.py $DATA_ROOT_DIR $OUTPUT_BASE_DIR

# generating Lidar accumulated depth
python $SCRIPTS_DIR/synchronization.py \
$DATA_ROOT_DIR \
$OUTPUT_BASE_DIR/Argoverse/gt_depth \
$OUTPUT_BASE_DIR/argoverse_ring_synchronized_data.pkl \
--acc_sweeps 5

# generating Lidar accumulated IP-basic post-processed depth
python $SCRIPTS_DIR/synchronization.py \
$DATA_ROOT_DIR \
$OUTPUT_BASE_DIR/Argoverse/gt_depth \
$OUTPUT_BASE_DIR/argoverse_ring_synchronized_data.pkl \
--acc_sweeps 5 \
--ip_basic

##############################################

# generating sparse Lidar projection all beams with no separation between up and down lidars
python $SCRIPTS_DIR/sparse_lidar_projection.py \
$DATA_ROOT_DIR \
$OUTPUT_BASE_DIR/Argoverse/sparse_lidar_projection \
$OUTPUT_BASE_DIR/argoverse_ring_synchronized_data.pkl

# generating sparse Lidar projection with 10 beams and only from up lidar
python $SCRIPTS_DIR/sparse_lidar_projection.py \
$DATA_ROOT_DIR \
$OUTPUT_BASE_DIR/Argoverse/sparse_lidar_projection \
$OUTPUT_BASE_DIR/argoverse_ring_synchronized_data.pkl \
--beams '0,1,2,3,4,5,6,8,12,14' \
--separate_pc 'up'