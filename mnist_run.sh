python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=entropy --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=cnn_distance --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17


python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=all --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
