import shutil
import os
import sqlite3
from score import *
from grit.lib.multiprocessing_utils import run_in_parallel

ScoreRecord = namedtuple(
    'ScoreRecord', 
    'factor sample principalId submission_date submission_fname bootstrap_index recall_at_10_fdr recall_at_50_fdr auPRC auROC rank'
)
def score_record_factory(cursor, row):
    row = list(row)
    row[3] = parse(row[3], fuzzy=True)
    return ScoreRecord(*row)

def calc_and_insert_new_results(
        DB, factor, sample, principalId, submission_date, submission_fname, score_callback, label_dir=''):

    labels_fname = os.path.join(label_dir, "{}.train.labels.tsv.gz".format(factor))
    # validate that the matching file exists and that it contains labels that 
    # match the submitted sample 
    header_data = None
    cell_line = sample
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


    all_res = []
    try:
        for bootstrap_i, results in calc_bootstrapped_scores(labels, scores):
            print(bootstrap_i, results)
            all_res.append([
                factor, 
                sample,

                principalId,
                submission_date,
                
                submission_fname,

                bootstrap_i,

                results.recall_at_10_fdr,
                results.recall_at_50_fdr,
                results.auPRC,
                results.auROC,
                
                -1
            ])
    except Exception as ex0:
        print("ERROR (calc_and_insert_new_results):", submission_fname, str(ex0))
        sys.stdout.flush()

    sys.stdout.flush()
   
    while True:
        try:
            conn = sqlite3.connect(DB)
            c = conn.cursor()
            for res in all_res:
                c.execute(
                    "INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                    res
                )
            c.close()
            conn.commit()
            conn.close()
        except sqlite3.OperationalError:
            conn.close()
            time.sleep(1)
            continue
        else:
            break
    
    print("END:",str(datetime.now()))
    return

def estimate_bootstrapped_scores_from_final_submissions_dir(
        DB, path, score_callback, label_dir, nthreads):
    conn = sqlite3.connect(DB)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS scores  (
        factor text,
        sample text,

        principalId int,
        submission_date text,
        submission_fname text,
        
        bootstrap_index int,

        recall_at_10_fdr real, 
        recall_at_50_fdr real, 
        auPRC real, 
        auROC real,
        
        rank int
    );''')
    c.close()
    conn.commit()
    conn.close()
    
    submission_args = []
    fnames_not_sorted = os.listdir(path)
    #for ff in fnames_not_sorted: print ff
    fnames = sorted(fnames_not_sorted, key=lambda x: x.split(".")[1], reverse=True)
    #for ff in fnames: print ff
    #os.exit(1)
    done = []
    for fname in os.listdir(path):
    # JIN
        #factor, sample, _, principal_id, _, _ = os.path.basename(fname).split(".")
    #['3343330', '7998005', 'F', 'NANOG', 'induced_pluripotent_stem_cell', 'tab', 'gz']
        principal_id, submissionId, _, factor, sample, _, _ = os.path.basename(fname).split(".")
        if (principal_id, factor, sample) in done:
            print("SKIPPING:", principal_id, factor, sample, fname)
            continue
        else:
            done.append( (principal_id, factor, sample) )

    #print os.path.basename(fname).split(".")

        full_fname = os.path.abspath(os.path.join(path, fname))
        #full_fname = os.path.basename( os.path.abspath(os.path.join(path, fname)) )
        #print factor, sample, principal_id, str(datetime.now()), full_fname
        submission_args.append([
            DB, 
            factor, sample, 
            principal_id, str(datetime.now()), full_fname,
            score_callback, label_dir
        ])

    run_in_parallel(nthreads, calc_and_insert_new_results, submission_args)
    return

def calc_combined_ranks(records):
    # make sure that the principalId is unique
    principal_ids = [x.principalId for x in records]
    # print "PIDs", len(principal_ids), len(set(principal_ids))
    submission_ids = [x.submission_fname.split(".")[1] for x in records]
    # submission_ids = [0 for x in records]

    attrs_to_rank = ['recall_at_10_fdr', 'recall_at_50_fdr', 'auPRC', 'auROC']
    scores = np.zeros(len(principal_ids), dtype=float)
    for user_i, attr in enumerate(attrs_to_rank):
        attr_scores = np.array([getattr(x, attr) for x in records])
        ranks = rankdata(-attr_scores, "average")
        # print attr, len(ranks)
        pval_scores = np.log(ranks/float(len(ranks) + 1))
        scores += pval_scores
    ranks = rankdata(scores, "average")
    # print ranks
    return dict(zip(zip(principal_ids, submission_ids), ranks))

def get_name(principalId):
    return principalId
    # if principalId > 100000000: 
    #     principalId -= 100000000 
    #     suffix = " (F)"
    # else:
    #     suffix = " (C)"
    # try: 
    #     res = syn.restGET('/team/{id}'.format(id=principalId))
    #     return res['name'] + suffix
    # except:
    #     profile = syn.getUserProfile(principalId)
    #     return profile['userName'] + suffix

GlobalScore = namedtuple('GlobalScore', ['principalId', 'name', 'score_lb', 'score_mean', 'score_ub', 'rank'])

def get_principalId(principalId):
    return principalId
    # if principalId > 100000000: return principalId-100000000
    # else: return principalId

def calculate_ranks_from_DB(DB, ids_to_keep=None):
    conn = sqlite3.connect(DB)
    conn.row_factory = score_record_factory
    c = conn.cursor()
    c.execute("SELECT * FROM scores ORDER BY bootstrap_index, principalId;")
    sample_grpd_results = defaultdict(lambda: defaultdict(list))
    all_users = set()
    for x in c.fetchall():
        # if ids_to_keep is set, filter by a subset of participants
        if ids_to_keep is not None and x.principalId not in ids_to_keep: 
            continue
        sample_key = (x.sample, x.factor)
        sample_grpd_results[(x.sample, x.factor)][x.bootstrap_index].append(x)
        all_users.add(x.principalId)
    
    # group all submissions by tf name and sample
    rv = {}
    global_scores = defaultdict(lambda: defaultdict(list))
    for (tf_name, sample), bootstrapped_submissions in sample_grpd_results.iteritems():
        if sample == 'ATF2': continue
        print("\n# %s %s Final Round" % (sample, tf_name))
        print("Syanpse ID | Team ID | Rank")
        print("---|---|---")
        user_ranks = defaultdict(list)
        for index, submissions in bootstrapped_submissions.iteritems():
            # ranks = calc_combined_ranks_final_1_and_2(submissions)
            ranks = calc_combined_ranks(submissions)

            obs_users = set(x[0] for x in ranks.keys())
            for (principalId, submission_id), rank in ranks.iteritems():
                # print principalId, rank
                # user_ranks[(principalId, submission_id)].append(rank)
                user_ranks[(principalId, 0)].append(rank)
                global_scores[index][principalId].append(
                    min(0.5, rank/(len(ranks)+1))
                )            
            for principalId in all_users - obs_users:
                global_scores[index][principalId].append(0.5)

        #for (principalId, submission_id), ranks in sorted(
        for (principalId, _), ranks in sorted(
                user_ranks.iteritems(), key=lambda x: sorted(x[1])[1]):
            print("%d | %s | %.2f" % (get_principalId(principalId), get_name(principalId), sorted(ranks)[1]))
            # print principalId, sorted(ranks)[1]
        print

    # group the scores by user
    user_grpd_global_scores = defaultdict(list)
    user_grpd_global_ranks = defaultdict(list)
    for bootstrap_index, bootstrap_global_scores in global_scores.iteritems():
        sorted_scores = sorted(
            bootstrap_global_scores.iteritems(), key=lambda x: sum(x[1]))
        ranks = rankdata([sum(x[1]) for x in sorted_scores])
        for (principalId, scores), rank in zip(sorted_scores, ranks):
            user_grpd_global_scores[principalId].append(sum(scores)/float(len(scores)))
            user_grpd_global_ranks[principalId].append(rank)
    global_data = []
    for principalId, scores in sorted(
            user_grpd_global_scores.iteritems(), key=lambda x: sum(x[1])):
        global_data.append(GlobalScore(*[
            principalId, get_name(principalId), 
            min(scores), sum(scores)/len(scores), max(scores), 
            sorted(user_grpd_global_ranks[principalId])[1]
        ]))
    global_data = sorted(global_data, key=lambda x: (x.rank, x.score_mean))
    # print("\t".join(
    #     ("name", "rank", "lb", "mean", "ub")))
    # for x in global_data: 
    #     print("%s\t%.2f\t%.2f\t%.2f\t%.2f" % (
    #         x.name, x.rank, x.score_lb, x.score_mean, x.score_ub))

    print("\n# Final Round Overall Results")
    print(" | ".join(("Team ID", "rank", "Lower bound", "Mean", "Upperbound")))
    print("|".join(("---",)*6))
    for x in global_data: 
        print("%s | %.2f | %.2f | %.2f | %.2f" % (
            x.name, x.rank, x.score_lb, x.score_mean, x.score_ub))

    return rv, global_data

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

def calc_bootstrapped_scores(labels, scores):
    from sklearn.cross_validation import StratifiedKFold
    for i, (indices, _) in enumerate(
            StratifiedKFold(labels, n_folds=10, random_state=0)):
        results = ClassificationResult(
            labels[indices], scores[indices].round(), scores[indices])
        yield i, results
    return

def main():
    parser = argparse.ArgumentParser(prog='Bootstrapped-scoring/ranking python script for ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge.',
        description='This script works with labels for FINAL ROUND only.')
    parser.add_argument('--submissions-dir', type=str,
                            help='A directory for FINAL ROUND submission files. File names should match with pattern F.[TF_NAME].[CELL_TYPE].gz.')
    parser.add_argument('--score-db-file', type=str,
                            help='Score database file (MySQL3 .db file). If defined, this script will skip scoring submissions and use scores in a DB file.')
    parser.add_argument('--label-dir', type=str,
                            help='Directory for label files of FINAL ROUNDS. Label files should have a format of [TF_NAME].train.labels.tsv.gz. Download from Files/Challenge Resources/scoring_script/labels/final.')
    parser.add_argument('--num-threads', type=int, default=4,
                            help='Number of threads to compute bootstrapped score.')
    args = parser.parse_args()

    if args.score_db_file:
        DB=args.score_db_file
    else:
        DB='score_final.db'
        if args.label_dir and args.submissions_dir:
            estimate_bootstrapped_scores_from_final_submissions_dir(
                DB,
                args.submissions_dir,
                score_final_main, args.label_dir, args.num_threads)
        else:
            raise Exception('You must specify valid directories for --label-dir and --submissions-dir.')

    if args.label_dir=='':
        raise Exception('You must specify a valid directory for --label-dir.')

    calculate_ranks_from_DB(DB)

if __name__ == '__main__':
    main()

