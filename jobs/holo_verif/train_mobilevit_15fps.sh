for seed in {0..5}
do
    echo calculating for seed $seed
    python train.py --config-name=valid_frame_classifier -m +experiment=valid_nonvalid/mobilevit15fps "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python calibration.py --config-name=valid_frame_classifier -m +experiment=valid_nonvalid/mobilevit15fps "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python test.py --config-name=valid_frame_classifier -m +experiment=valid_nonvalid/mobilevit15fps "paths.split_name=k0,k1,k2,k3,k4" seed=$seed
done