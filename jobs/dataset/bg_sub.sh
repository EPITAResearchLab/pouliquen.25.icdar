
find data/midvdynattack/cropped/fraud/ -maxdepth 3 -mindepth 3 -type d -exec -i python scripts/bg_sub.py --input {} \;
find data/midvdynattack/cropped/origins/ -maxdepth 2 -mindepth 2 -type d -exec -i python scripts/bg_sub.py --input {} \;