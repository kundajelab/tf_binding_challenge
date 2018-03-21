import os, sys
import re
from itertools import izip_longest
import random
import math
from bisect import bisect
from collections import namedtuple, defaultdict
import gzip
import json
from dateutil.parser import parse
import time
from datetime import datetime


import sqlite3

import numpy as np
import pandas
from scipy.stats import itemfreq, norm, rankdata

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc

# needed because of an rpy2 linking bug
import readline
import rpy2
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
import rpy2.robjects as ro

import synapseclient
from synapseclient import Evaluation, Submission, SubmissionStatus
import challenge_config as conf

USER = ''
PASS = ''

#FINAL_LABELS_BASEDIR = "/mnt/data/TF_binding/DREAM_challenge/hidden_test_set_chipseq_data/tsvs/"
#VALIDATION_LABELS_BASEDIR = "/mnt/data/leaderboard_labels_w_chr8/"
#FINAL_LABELS_BASEDIR = "/mnt/data/hidden_test_set_chipseq_data/"
#FINAL_LABELS_BASEDIR = "/mnt/data/hidden_test_set_chipseq_data_whole_genome/"
FINAL_LABELS_BASEDIR = "/mnt/data/TF_binding/DREAM_challenge/hidden_test_set_chipseq_data/tsvs" # whole genome
FINAL_WITHIN_LABELS_BASEDIR = "/mnt/data/final_within_fake_lables/" # whole genome
VALIDATION_LABELS_BASEDIR = "/mnt/data/leaderboard_labels_w_chr8/"
#DB = "/encode/final2/bootrapped_scores.db"
#SUBMISSIONS_DIR = "/encode/final2/submissions/"
#SUBMISSIONS_DIR = "/encode/final2/test_submissions/"

# DB = "/srv/scratch/leepc12/DREAM_challenge/final2/filtered_bootstrapped_scores.db"
# DB = "/users/leepc12/code/EncodeChallenge/final2/final2_true.db"
# DB = "/users/leepc12/code/EncodeChallenge/final2/final1_true.db"
# DB = "/users/leepc12/code/EncodeChallenge/final2/final_1_and_2_with_10_fold.db"

#DB = "/users/leepc12/code/EncodeChallenge/final2/final_1_and_2_with_10_fold_final2_participants_have_high_synapseId.db"
DB = "final2/final_1_and_2_with_10_fold_final2_participants_have_high_synapseId.db"

SUBMISSIONS_DIR = "/srv/scratch/leepc12/DREAM_challenge/final2/filtered_submissions/"

# DB = "/mnt/data/TF_binding/DREAM_challenge/final1/bootstrapped_scores.db"
# SUBMISSIONS_DIR = "/mnt/data/TF_binding/DREAM_challenge/final1/test_submissions/"


syn = synapseclient.Synapse()
syn.login(email=USER, password=PASS)

def optional_gzip_open(fname):
    return gzip.open(fname) if fname.endswith(".gz") else open(fname)  

# which measures to use to evaluate the overall score
MEASURE_NAMES = ['recall_at_10_fdr', 'recall_at_50_fdr', 'auPRC', 'auROC']
ValidationResults = namedtuple('ValidationResults', MEASURE_NAMES)

# match filenames of the form (optional.part.){L,F,B}.[TF].[cell-line].tab.gz
fname_pattern = "[0-9\.]*([LFB])\.(.+?)\.(.+?).tab(?:\(\d+\))?.gz$"

def recall_at_fdr(y_true, y_score, fdr_cutoff=0.05):
    print(y_true, y_score)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    fdr = 1- precision
    cutoff_index = next(i for i, x in enumerate(fdr) if x <= fdr_cutoff)
    return recall[cutoff_index]

def scikitlearn_calc_auPRC(y_true, y_score):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return auc(recall, precision)

def calc_auPRC(y_true, y_score):
    """Calculate auPRC using the R package 

    """
    ro.globalenv['pred'] = y_score
    ro.globalenv['labels'] = y_true
    return ro.r('library(PRROC); pr.curve(scores.class0=pred, weights.class0=labels)$auc.davis.goadrich')[0]

class InputError(Exception):
    pass

ClassificationResultData = namedtuple('ClassificationResult', [
    'is_cross_celltype',
    'sample_type', # should be validation or test
    'train_chromosomes',
    'train_samples', 

    'validation_chromosomes',
    'validation_samples', 

    'auROC', 'auPRC', 'F1', 
    'recall_at_50_fdr', 'recall_at_25_fdr', 
    'recall_at_10_fdr', 'recall_at_05_fdr',
    'num_true_positives', 'num_positives',
    'num_true_negatives', 'num_negatives'])

class ClassificationResult(object):
    _fields = ClassificationResultData._fields

    def __iter__(self):
        return iter(getattr(self, field) for field in self._fields)

    def iter_items(self):
        return zip(self._fields, iter(getattr(self, field) for field in self._fields))
    
    def __init__(self, labels, predicted_labels, predicted_prbs,
                 is_cross_celltype=None, sample_type=None,
                 train_chromosomes=None, train_samples=None,
                 validation_chromosomes=None, validation_samples=None):
        # filter out ambiguous labels
        # JIN
        index = labels > -0.5
        predicted_labels = predicted_labels[index]
        predicted_prbs = predicted_prbs[index]
        labels = labels[index]


        self.is_cross_celltype = is_cross_celltype
        self.sample_type = sample_type

        self.train_chromosomes = train_chromosomes
        self.train_samples = train_samples

        self.validation_chromosomes = validation_chromosomes
        self.validation_samples = validation_samples
        
        positives = np.array(labels == 1)
        self.num_true_positives = (predicted_labels[positives] == 1).sum()
        self.num_positives = positives.sum()
        
        negatives = np.array(labels == 0)        
        self.num_true_negatives = (predicted_labels[negatives] == 0).sum()
        self.num_negatives = negatives.sum()

        if positives.sum() + negatives.sum() < len(labels):
            raise InputError("All labels must be either 0 or +1")
        
        try: self.auROC = roc_auc_score(positives, predicted_prbs)
        except ValueError: self.auROC = float('NaN')
        self.auPRC = calc_auPRC(positives, predicted_prbs)
        self.F1 = f1_score(positives, predicted_labels)
        self.recall_at_50_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.50)
        self.recall_at_25_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.25)
        self.recall_at_10_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.10)
        self.recall_at_05_fdr = recall_at_fdr(
            labels, predicted_prbs, fdr_cutoff=0.05)
        return

    @property
    def positive_accuracy(self):
        return float(self.num_true_positives)/(1e-6 + self.num_positives)

    @property
    def negative_accuracy(self):
        return float(self.num_true_negatives)/(1e-6 + self.num_negatives)

    @property
    def balanced_accuracy(self):
        return (self.positive_accuracy + self.negative_accuracy)/2    

    def iter_numerical_results(self):
        for key, val in self.iter_items():
            try: _ = float(val) 
            except TypeError: continue
            yield key, val
        return

    def __str__(self):
        rv = []
        if self.train_samples is not None:
            rv.append("Train Samples: %s\n" % self.train_samples)
        if self.train_chromosomes is not None:
            rv.append("Train Chromosomes: %s\n" % self.train_chromosomes)
        if self.validation_samples is not None:
            rv.append("Validation Samples: %s\n" % self.validation_samples)
        if self.validation_chromosomes is not None:
            rv.append("Validation Chromosomes: %s\n" % self.validation_chromosomes)
        rv.append("Bal Acc: %.3f" % self.balanced_accuracy )
        rv.append("auROC: %.3f" % self.auROC)
        rv.append("auPRC: %.3f" % self.auPRC)
        rv.append("F1: %.3f" % self.F1)
        rv.append("Re@0.50 FDR: %.3f" % self.recall_at_50_fdr)
        rv.append("Re@0.25 FDR: %.3f" % self.recall_at_25_fdr)
        rv.append("Re@0.10 FDR: %.3f" % self.recall_at_10_fdr)
        rv.append("Re@0.05 FDR: %.3f" % self.recall_at_05_fdr)
        rv.append("Positive Accuracy: %.3f (%i/%i)" % (
            self.positive_accuracy, self.num_true_positives,self.num_positives))
        rv.append("Negative Accuracy: %.3f (%i/%i)" % (
            self.negative_accuracy, self.num_true_negatives, self.num_negatives))
        return "\t".join(rv)


def build_sample_test_file(truth_fname, score_column_index, output_fname):
    ofp = open(output_fname, "w")
    with optional_gzip_open(truth_fname) as fp:
        for i, line in enumerate(fp):
            # skip the header
            if i == 0: continue
            if i%1000000 == 0: print("Finished processing line", i)
            data = line.split()
            region = data[:3]
            label = data[score_column_index]
            if label == 'U':
                score = random.uniform(0, 0.8)
            elif label == 'A':
                score = random.uniform(0.5, 0.9)
            elif label == 'B':
                score = random.uniform(0.7, 1.0)
            data = region + [score,]
            ofp.write("{}\t{}\t{}\t{}\n".format(*data))
    ofp.close()
    return 

def verify_file_and_build_scores_array(
        truth_fname, submitted_fname, labels_index):
    # make sure the region entries are identical and that
    # there is a float for each entry
    truth_fp_iter = iter(optional_gzip_open(truth_fname))
    submitted_fp_iter = iter(optional_gzip_open(submitted_fname))

    # skip the header
    next(truth_fp_iter)
    
    scores = []
    labels = []
    t_line_num, s_line_num, s_scored_line_num = 0, 0, 0
    while True:
        # get the next line
        try:
            t_line = next(truth_fp_iter)
            t_line_num += 1
            s_line = next(submitted_fp_iter)
            s_line_num += 1
            s_scored_line_num += 1
        except StopIteration:
            break
        
        # parse the truth line
        t_match = re.findall("(\S+\t\d+\t\d+)\t(.+?)\n", t_line)
        assert len(t_match) == 1, \
            "Line %i in the labels file did not match the expected pattern '(\S+\t\d+\t\d+)\t(.+?)\n'" % t_line_num
        
        # parse the submitted file line, raising an error if it doesn't look
        # like expected
        s_match = re.findall("(\S+\t\d+\t\d+)\t(\S+)\n", s_line)
        if len(s_match) != 1:
            raise InputError("Line %i in submitted file does not conform to the required pattern: '(\S+\t\d+\t\d+)\t(\S+)\n'" 
                             % t_line_num)
        if t_match[0][0] != s_match[0][0]:
            raise InputError("Line %i in submitted file does not match line %i in the reference regions file" 
                             % (t_line_num, s_line_num))

        # parse and validate the score
        try:
            score = float(s_match[0][1])
        except ValueError:
            raise InputError("The score at line %i in the submitted file can not be interpreted as a float" % s_line_num)
        scores.append(score)

        # add the label
        region_labels = t_match[0][-1].split()
        assert all(label in 'UAB' for label in region_labels), "Unrecognized label '{}'".format(
            region_labels)
        region_label = region_labels[labels_index]
        if region_label == 'A':
            labels.append(-1)
        elif region_label == 'B':
            labels.append(1)
        elif region_label == 'U':
            labels.append(0)
        else:
            assert False, "Unrecognized label '%s'" % region_label

    # make sure the files have the same number of lines
    if t_line_num < s_scored_line_num:
        raise InputError("The submitted file has more rows than the reference file")
    if t_line_num > s_scored_line_num:
        raise InputError("The reference file has more rows than the reference file")
    
    return np.array(labels), np.array(scores)

def build_ref_scores_array(ref_fname):
    factor = os.path.basename(ref_fname).split(".")[0]
    
    with gzip.open(ref_fname) as fp:
        header_data = fp.readline().split()
        sample_names = header_data[3:]
    samples = {}
    df = pandas.read_csv(ref_fname, sep="\t")
    # filter the data frame by the test regions
    contig_mask = ( (df['chr'].values == 'chr1') 
                    | (df['chr'].values == 'chr8') 
                    | (df['chr'].values == 'chr21') )
    # JIN
    for sample_name in sample_names:
        char_data = df[sample_name].values[contig_mask]
        samples[sample_name] = np.zeros(char_data.shape, dtype=int)
        samples[sample_name][char_data == 'B'] = 1
        samples[sample_name][char_data == 'A'] = -1
    return samples

def build_submitted_scores_array(fname):
    df = pandas.read_csv(
        fname, names=["chr", "start", "stop", "sample"], sep="\t")
    return df['sample'].values

def verify_and_score_submission(ref_fname, submitted_fname,label_index):
    # build the label and score matrices
    labels, scores = verify_file_and_build_scores_array(
        ref_fname, submitted_fname, label_index)

    # compute the measures on the full submitted scores
    full_results = ClassificationResult(labels, scores.round(), scores)
    results = ValidationResults(*[
        getattr(full_results, attr_name) for attr_name in MEASURE_NAMES])

    return results, labels, scores

def test_main():
    # labels_file = sys.argv[1]
    # build_sample_test_file(labels_file, 3, "test.scores")
    # submission_file = "test.scores"
    labels_file = "/mnt/data/TF_binding/DREAM_challenge/hidden_test_set_chipseq_data/tsvs/JUND.train.labels.tsv.gz"
    submission_file="/srv/scratch/leepc12/DREAM_challenge/final2/submissions_failed/3342173.8012980.F.JUND.liver.tab.gz"
    cell_line="liver"

    header_data = None
    with gzip.open(labels_file) as fp:
        header_data = next(fp).split()
    # Make sure the header looks right
    assert header_data[:3] == ['chr', 'start', 'stop']
    labels_file_samples = header_data[3:]

    label_index = labels_file_samples.index(cell_line)
    # labels, scores = verify_file_and_build_scores_array(
    #     labels_file, submission_file, label_index)

    results, labels, scores = verify_and_score_submission(labels_file, submission_file,label_index)
    calc_bootstrapped_scores( labels, scores )
    print(results)

def score_two_files_main():
    try:
        labels_fname = sys.argv[1]
        submission_fname = sys.argv[2]
    except IndexError:
        print("usage: python score.py labels_fname submission_fname")
        sys.exit(0)
    labels, scores = verify_file_and_build_scores_array(
        labels_fname, submission_fname)
    full_results = ClassificationResult(labels, scores.round(), scores)
    print(full_results)

def score_main(submission_fname):
    # load and parse the submitted filename
    res = re.findall(fname_pattern, os.path.basename(submission_fname))
    if len(res) != 1 or len(res[0]) != 3:
        raise InputError("The submitted filename ({}) does not match expected naming pattern '{}'".format(
            submission_fname, fname_pattern))
    else:
        submission_type, factor, cell_line = res[0]
    
    # find a matching validation file
    labels_fname = os.path.join(VALIDATION_LABELS_BASEDIR, "{}.labels.tsv.gz".format(factor))
    # validate that the matching file exists and that it contains labels that 
    # match the submitted sample 
    header_data = None
    try:
        with gzip.open(labels_fname) as fp:
            header_data = next(fp).split()
    except IOError:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid leaderboard submission.".format(
            factor, cell_line))

    # Make sure the header looks right
    assert header_data[:3] == ['chr', 'start', 'stop']
    labels_file_samples = header_data[3:]
    # We only expect to see one sample per leaderboard sample
    if cell_line not in labels_file_samples:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid leaderboard submission.".format(
            factor, cell_line))

    label_index = labels_file_samples.index(cell_line)
    labels, scores = verify_file_and_build_scores_array(
        labels_fname, submission_fname, label_index)
    full_results = ClassificationResult(labels, scores.round(), scores)
    return full_results, labels, scores

def calc_bootstrapped_scores(labels, scores):
    from sklearn.cross_validation import StratifiedKFold
    for i, (indices, _) in enumerate(
            StratifiedKFold(labels, n_folds=10, random_state=0)):
        results = ClassificationResult(
            labels[indices], scores[indices].round(), scores[indices])
        yield i, results
    return


def score_final_main(submission_fname):
    # load and parse the submitted filename
    # 3343330.7278071.1474630242.F.NANOG.induced_pluripotent_stem_cell.tab.gz
    fname_pattern = "(.+?)\.(.+?)\.F\.(.+?)\.(.+?)\.tab.gz"
    #fname_pattern = "(.+?)\.(.+?)\.FSUBSET\.(\d+)\.tab.gz"
    res = re.findall(fname_pattern, os.path.basename(submission_fname))
    #print res, len(res), len(res[0])
    #if len(res) != 1 or len(res[0]) != 3:
    if len(res) != 1 or len(res[0]) != 4:
        raise InputError("The submitted filename ({}) does not match expected naming pattern '{}'".format(
            submission_fname, fname_pattern))
    else:
        #factor, cell_line, team_id = res[0]
        teamid, _, factor, cell_line = res[0]
    # find a matching validation file
    labels_fname = os.path.join(FINAL_LABELS_BASEDIR, "{}.train.labels.tsv.gz".format(factor))
    # validate that the matching file exists and that it contains labels that 
    # match the submitted sample 
    header_data = None
    try:
        with gzip.open(labels_fname) as fp:
            header_data = next(fp).split()
    except IOError:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))
    # Make sure the header looks right
    assert header_data[:3] == ['chr', 'start', 'stop']
    labels_file_samples = header_data[3:]
    # We only expect to see one sample per leaderboard sample
    if cell_line not in labels_file_samples:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))

    try:
        scores = build_submitted_scores_array(submission_fname)
        labels = build_ref_scores_array(labels_fname)[cell_line]
    # JIN
        full_results = ClassificationResult(labels, scores.round(), scores)
    except:
        print("ERROR", submission_fname)
        sys.stdout.flush()

    return full_results, labels, scores

def score_final_within_celltype_main(submission_fname):
    # load and parse the submitted filename
    # 3343330.7278071.1474630242.F.NANOG.induced_pluripotent_stem_cell.tab.gz
    fname_pattern = "F\.(.+?)\.(.+?)\.tab.gz"
    #fname_pattern = "(.+?)\.(.+?)\.FSUBSET\.(\d+)\.tab.gz"
    res = re.findall(fname_pattern, os.path.basename(submission_fname))
    #print res, len(res), len(res[0])
    if len(res) != 1 or len(res[0]) != 3:
    #if len(res) != 1 or len(res[0]) != 4:
        raise InputError("The submitted filename ({}) does not match expected naming pattern '{}'".format(
            submission_fname, fname_pattern))
    else:
        #factor, cell_line, team_id = res[0]
        teamid, _, factor, cell_line = res[0]
    # find a matching validation file
    # labels_fname = os.path.join(FINAL_LABELS_BASEDIR, "{}.train.labels.tsv.gz".format(factor))
    labels_fname = os.path.join(FINAL_WITHIN_LABELS_BASEDIR, "{}.train.labels.tsv.gz".format(factor))    
    # validate that the matching file exists and that it contains labels that 
    # match the submitted sample 
    header_data = None
    try:
        with gzip.open(labels_fname) as fp:
            header_data = next(fp).split()
    except IOError:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))
    # Make sure the header looks right
    assert header_data[:3] == ['chr', 'start', 'stop']
    labels_file_samples = header_data[3:]
    # We only expect to see one sample per leaderboard sample
    if cell_line not in labels_file_samples:
        raise InputError("The submitted factor, sample combination ({}, {}) is not a valid final submission.".format(
            factor, cell_line))

    scores = build_submitted_scores_array(submission_fname)
    labels = build_ref_scores_array(labels_fname)[cell_line]
    # JIN
    full_results = ClassificationResult(labels, scores.round(), scores)
    return full_results, labels, scores

def main():
    try:
        submission_fname = sys.argv[1]
    except IndexError:
        print("usage: python score.py submission_fname")
        sys.exit(0)
    results, labels, scores = score_main(submission_fname)
    print(results)

if __name__ == '__main__':
    #main()
    test_main()
