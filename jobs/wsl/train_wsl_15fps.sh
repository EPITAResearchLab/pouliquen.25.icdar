for seed in {0..5}
do
    echo calculating for seed $seed
    python train.py --config-name=wsl -m +experiment=wsl/mobilevit_s_15fps_old2 "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python calibration.py --config-name=wsl -m +experiment=wsl/mobilevit_s_15fps_old2 "paths.split_name=k0,k1,k2,k3,k4" seed=$seed

    python test.py --config-name=wsl -m +experiment=wsl/mobilevit_s_15fps_old2 "paths.split_name=k0,k1,k2,k3,k4" seed=$seed
    echo "Done for seed $seed"
done