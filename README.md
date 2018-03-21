# tf_binding_challenge
scoring/ranking code for tf binding challenge

1) add your synapse auth info to score.py (line 34)
```
USER=''
PASS=''
```

2) install conda: https://conda.io/miniconda.html

3) install dependencies
```
conda create -n tf_binding_challenge python==2.7.6
source activate tf_binding_challenge
conda install pandas
conda install scikit-learn
conda install rpy2
pip install synapseclient

git clone https://github.com/nboley/grit
cd grit
python setup.py install
```

4) run rank.py
```
source activate tf_binding_challenge
python rank.py
```

