for seed in {0..5}
do
    echo calculating for seed $seed
    python train.py --config-name=classifier -m +experiment=classifier/mobilevit_s_5fps_old_sub "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python calibration.py --config-name=classifier -m +experiment=classifier/mobilevit_s_5fps_old_sub "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python test.py --config-name=classifier -m +experiment=classifier/mobilevit_s_5fps_old_sub "paths.split_name=k0,k1,k2,k3,k4" seed=$seed
done
echo done