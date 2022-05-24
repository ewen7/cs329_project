python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --feature-distribution 0.1 0.1 0.1 0.1 0.18 0.1 0.1 0.1 0.1 0.02
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --al-method=entropy --feature-distribution 0.1 0.1 0.1 0.1 0.18 0.1 0.1 0.1 0.1 0.02
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters-10 --model=cnn --al-method=cnn_distance --feature-distribution 0.1 0.1 0.1 0.1 0.18 0.1 0.1 0.1 0.1 0.02
