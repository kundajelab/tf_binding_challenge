# Scoring/ranking script for ENCODE DREAM TF binding site challenge

## Installation

scoring/ranking code for ENCODE DREAM TF binding site challenge

1) Install conda: https://conda.io/miniconda.html
2) Install MySQL3
```
$ sudo apt-get install sqlite3 libsqlite3-dev
```

3) Install Conda dependencies.
```
$ bash install_dependencies.sh
```

4) Make a directory `[LABEL_DIR]` for label files.
5) Download label files from [here](https://www.synapse.org/#!Synapse:syn10164048) on `[LABEL_DIR]`.

6) ACTIVATE CONDA ENVIRONMENT BEFORE RUNNING SCORING/RANKING SCRIPTS!
```
$ source activate tf_binding_challenge
```

## Scoring script
Scoring script takes in two positional parameters.
```
$ python score.py [SUBMISSION_FILE_NAME] [LABEL_DIR]
```

## Ranking script
There are two steps for the ranking script.

1) Calculate bootstrapped for each submission file and generate a MySQL3 database file `score_final.db`. This will take time. Give an good number of thread `--num-threads` to this job (default=4).
```
$ python rank.py --submissions-dir [SUBMISSION_DIR] --label-dir [LABEL_DIR] --num-threads 4
```

SUBMISSION FILE NAMES IN `[SUBMISSION_DIR]` MUST MATCH THE FOLLOWING PATTERN. Any unique integer will be okay for `[TEAM_ID]` and `[SUBMISSION_ID]`.
```
[TEAM_ID].[SUBMISSION_ID].F.[TF].[CELL_LINE].tab.gz
```

Example file names in `[SUBMISSION_DIR]`.
```
3319559.8006821.F.CTCF.PC-3.tab.gz
3319559.8006822.F.CTCF.induced_pluripotent_stem_cell.tab.gz
3319559.8006823.F.E2F1.K562.tab.gz
3319559.8006824.F.EGR1.liver.tab.gz
3319559.8006827.F.FOXA1.liver.tab.gz
3319559.8006832.F.FOXA2.liver.tab.gz
3319559.8006833.F.GABPA.liver.tab.gz
3319559.8006834.F.HNF4A.liver.tab.gz
3319559.8006835.F.JUND.liver.tab.gz
3319559.8006836.F.MAX.liver.tab.gz
3319559.8006837.F.NANOG.induced_pluripotent_stem_cell.tab.gz
3319559.8006838.F.REST.liver.tab.gz
3319559.8006839.F.TAF1.liver.tab.gz
3321063.8001182.F.REST.liver.tab.gz
3321063.8001193.F.CTCF.PC-3.tab.gz
3321063.8001208.F.CTCF.induced_pluripotent_stem_cell.tab.gz
3321063.8001217.F.EGR1.liver.tab.gz
3321063.8001222.F.FOXA1.liver.tab.gz
3321063.8001234.F.MAX.liver.tab.gz
3321063.8001277.F.GABPA.liver.tab.gz
3321063.8002362.F.TAF1.liver.tab.gz
3321063.8003251.F.FOXA2.liver.tab.gz
...
```

2) Calculate ranking based on the database file `score_final.db`. You can also skip step 1 and start from step 2 because step 1 generates a database file `score_final.db`.
```
$ python rank.py --score-db-file [DB_FILE_NAME]
```
