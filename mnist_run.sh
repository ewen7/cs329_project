# python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
# python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=entropy --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
# python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=cnn_distance --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17


python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=penguin --al-method=all --feature-distribution 0.05 0.12 0.04 0.05 0.02 0.11 0.13 0.14 0.17 0.17
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=dodo --al-method=all --feature-distribution 0.18 0.02 0.18 0.02 0.18 0.02 0.18 0.02 0.18 0.02
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=cassowary --al-method=all --feature-distribution 0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.17 0.19
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=emu --al-method=all --feature-distribution 0.19 0.17 0.15 0.13 0.11 0.09 0.07 0.05 0.03 0.01
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=ostrich --al-method=all --feature-distribution 0.15 0.05 0.15 0.05 0.15 0.05 0.15 0.05 0.15 0.05
python exp.py --dataset='mnist' --dataset-split=0.01 --al-proposal-size=300 --al-iters=10 --model=cnn --name=kiwi --al-method=all --feature-distribution 0.2 0.05 0.03 0.13 0.09 0.04 0.22 0.1 0.1 0.04
