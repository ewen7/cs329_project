# python exp.py --dataset='hdp' --dataset-split=1 --al-iters=0 --verbose
# python exp.py --dataset='hdp' --dataset-split=0.01 --al-iters=3 --al-proposal-size=50 --al-method='random' --verbose
# python exp.py --dataset='hdp' --dataset-split=0.01 --al-iters=3 --al-proposal-size=50 --al-method='entropy' --verbose
python exp.py --dataset='hdp' --dataset-split=1 --al-iters=0 --verbose
python exp.py --dataset='hdp' --dataset-split=0.02 --al-iters=3 --al-proposal-size=500 --al-method='random' 
python exp.py --dataset='hdp' --dataset-split=0.02 --al-iters=3 --al-proposal-size=500 --al-method='entropy'
