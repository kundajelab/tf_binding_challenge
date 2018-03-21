import shutil
from score import *
from grit.lib.multiprocessing_utils import run_in_parallel
#NTHREADS = 8
NTHREADS = 20

def iter_grpd_submissions(submission_queue_id):
    grpd_submissions = defaultdict(lambda: defaultdict(list))
    for submission, status in syn.getSubmissionBundles(
            submission_queue_id, status='VALIDATED'):
        #print(submission.id, status)
        # skip unscored submissions
        #if status['status'] != 'VALIDATED': continue
        principalId = submission.teamId if 'teamId' in submission else submission.userId
        creation_ts = parse(submission['createdOn'], fuzzy=True)
        file_handles = [
            x for x in json.loads(submission['entityBundleJSON'])['fileHandles']
            if x['concreteType'] != u'org.sagebionetworks.repo.model.file.PreviewFileHandle' 
        ]
        assert len(file_handles) == 1, str([x['fileName'] for x in file_handles])
        submission_fname = file_handles[0]['fileName']
        if submission_fname == 'NOT_SET':
            print("Skipping: %s" % submission_fname)
            continue
        submission_round, factor, sample = submission_fname.split('.')[0:3]
        # skip final round submissions
        #if submission_round == 'F': continue
        #assert submission_round == 'L'
        filename = "{}/{}.{}.{}".format(SUBMISSIONS_DIR, principalId, submission.id, submission_fname)
        if not os.path.isfile(filename): 
            print("Downloading submission: {}".format(submission.id))
            submission = syn.getSubmission(
                submission, 
                downloadLocation="/encode/final_cache/", 
            )
            assert not os.path.isfile(filename)
            print("Copying {} to {}".format(submission.filePath, filename))
            shutil.copy(submission.filePath, filename)
            #print "Skipping: %s" % filename
            #continue
        grpd_submissions[
            (factor, sample)][
                principalId].append((creation_ts, filename))
    
    for leader_board, factor_submissions in grpd_submissions.iteritems():
        yield leader_board, factor_submissions

    return

ScoreRecord = namedtuple(
    'ScoreRecord', 
    'factor sample principalId submission_date submission_fname bootstrap_index recall_at_10_fdr recall_at_50_fdr auPRC auROC rank'
)
def score_record_factory(cursor, row):
    row = list(row)
    row[3] = parse(row[3], fuzzy=True)
    return ScoreRecord(*row)

def calc_and_insert_new_results(
        DB, factor, sample, principalId, submission_date, submission_fname, score_callback):
    # sort by submission date
    #print (factor, sample), principalId, submission_date, submission_fname, str(datetime.now())
    #print "START:",str(datetime.now())
    sys.stdout.flush()
    try:
        full_results, labels, scores = score_callback(submission_fname)
    except:
        print("ERROR (calc_and_insert_new_results, full):", submission_fname)
    sys.stdout.flush()

    all_res = []
 #    all_res.append([
 #        factor,
 #        sample,

 #        principalId,
 #        submission_date,

 #        submission_fname,

 #        -1,

 #        full_results.recall_at_10_fdr,
 #        full_results.recall_at_50_fdr,
 #        full_results.auPRC,
 #        full_results.auROC,

 #        -1
 #    ])
    print("FULL", principalId, submission_fname, full_results)

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
        print("ERROR (calc_and_insert_new_results):", submission_fname, bootstrap_i, str(ex0))
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

def estimate_bootstrapped_scores_from_submission_queue(
        DB, submission_queue_id, score_callback):
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
    for (factor, sample), factor_submissions in iter_grpd_submissions(
            submission_queue_id):
        for principalId, submissions in factor_submissions.iteritems():
            submissions.sort(reverse=True)
            for (submission_index, 
                 (submission_date, submission_fname)) in enumerate(submissions):
                # skip old submissions
                if submission_index > 0: continue
                conn = sqlite3.connect(DB)
                c = conn.cursor()
                c.execute(
                    "SELECT * FROM scores WHERE factor=? AND sample=? AND principalId=? AND submission_date=?",
                    (
                        factor,
                        sample,
                        principalId,
                        submission_date
                    )
                )
                res = c.fetchall()
                c.close()
                conn.close()
                if len(res) == 0:
                    submission_args.append([
                        DB, 
                        factor, sample, 
                        principalId, submission_date, submission_fname,
                        score_callback
                    ])

    run_in_parallel(NTHREADS, calc_and_insert_new_results, submission_args)
    return

def estimate_bootstrapped_scores_from_final_submissions_dir(
        DB, path, score_callback):
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
            score_callback
        ])

    run_in_parallel(NTHREADS, calc_and_insert_new_results, submission_args)
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

def calc_combined_ranks_final_1_and_2(records):
    # make sure that the principalId is unique
    principal_ids = [x.principalId for x in records]
    # submission_ids = [x.submission_fname.split(".")[1] for x in records]

    attrs_to_rank = ['recall_at_10_fdr', 'recall_at_50_fdr', 'auPRC', 'auROC']
    scores = np.zeros(len(principal_ids), dtype=float)
    for user_i, attr in enumerate(attrs_to_rank):
        attr_scores = np.array([getattr(x, attr) for x in records])
        ranks = rankdata(-attr_scores, "average")
        # print attr, len(ranks)
        pval_scores = np.log(ranks/float(len(ranks) + 1))
        scores += pval_scores
    ranks = rankdata(scores, "average")

    # print "ranks (before)", len(ranks), len(principal_ids), list(zip(principal_ids,ranks))
    
    new_records = []
    principal_ids_already_taken = set()
    for i, x in enumerate(records):
        if x.principalId in principal_ids_already_taken:
            continue
        ranks_of_dupes = [(j,ranks[j]) for j in range(len(principal_ids)) if principal_ids[j]==x.principalId]
        lowest_id = sorted(ranks_of_dupes, key=lambda z: z[1])[0][0]
        # print x.principalId, lowest_id, principal_ids[lowest_id], ranks_of_dupes
        new_records.append( records[lowest_id] )
        for (j,_) in ranks_of_dupes:
            principal_ids_already_taken.add( principal_ids[j] )

    new_principal_ids = [x.principalId for x in new_records]
    scores = np.zeros(len(new_principal_ids), dtype=float)
    for user_i, attr in enumerate(attrs_to_rank):
        attr_scores = np.array([getattr(x, attr) for x in new_records])
        ranks = rankdata(-attr_scores, "average")
        # print attr, len(ranks)
        pval_scores = np.log(ranks/float(len(ranks) + 1))
        scores += pval_scores
    ranks = rankdata(scores, "average")

    # print "ranks (after)", len(ranks), len(new_principal_ids), list(zip(new_principal_ids,ranks))
    # print "ranks:", ranks
    # print "scores:", scores
    
    # print len(principal_ids), len(new_principal_ids)
    new_submission_ids = [0 for x in new_records]
    return dict(zip(zip(new_principal_ids, new_submission_ids), ranks))

def filter_older_submissions(submissions):
    """Choose the most recent submission for each user.   

    """
    filtered_submissions = {}
    for submission in submissions:
        if (submission.principalId not in filtered_submissions
            or (filtered_submissions[submission.principalId].submission_date 
                < submission.submission_date)
            ):
            filtered_submissions[submission.principalId] = submission
    return filtered_submissions.values()

def get_name(principalId):    
    if principalId > 100000000: 
        principalId -= 100000000 
        suffix = " (F)"
    else:
        suffix = " (C)"
    try: 
        res = syn.restGET('/team/{id}'.format(id=principalId))
        return res['name'] + suffix
    except:
        profile = syn.getUserProfile(principalId)
        return profile['userName'] + suffix

GlobalScore = namedtuple('GlobalScore', ['principalId', 'name', 'score_lb', 'score_mean', 'score_ub', 'rank'])

def get_principalId(principalId):
    if principalId > 100000000: return principalId-100000000
    else: return principalId

def build_ranks_spreadsheet(DB):
    conn = sqlite3.connect(DB)
    conn.row_factory = score_record_factory
    c = conn.cursor()
    c.execute("SELECT * FROM scores ORDER BY bootstrap_index, principalId;")
    sample_grpd_results = defaultdict(lambda: defaultdict(list))
    all_users = set()
    for x in c.fetchall():
        sample_key = (x.sample, x.factor)
        sample_grpd_results[(x.sample, x.factor)][x.bootstrap_index].append(x)
        all_users.add(x.principalId)
    
    # group all submissions by tf name and sample
    rv = {}
    global_scores = defaultdict(lambda: defaultdict(list))
    for (tf_name, sample), bootstrapped_submissions in sample_grpd_results.iteritems():
        # print tf_name, sample
        if sample == 'ATF2': continue
        print("# %s %s Final Round Score" % (sample, tf_name))
        print("Synapse ID | Team name | auROC | auPRC | Re@50% FDR | Re@10% FDR")
        print("---|---|---|---|---|---")
        data = []
        for i, x in enumerate(
                sorted(bootstrapped_submissions[0], key = lambda x: -x.auPRC)):
            # data.append((x.principalId, 
            data.append((get_principalId(x.principalId),
                         get_name(x.principalId), 
                         x.auROC, 
                         x.auPRC, 
                         x.recall_at_50_fdr, 
                         x.recall_at_10_fdr))
            # print("\t".join(str(x) for x in data[-1]))
            print(" | ".join(str(x) for x in data[-1]))
            #print x.principalId, get_name(x.principalId).ljust(30), "%.4f   "%x.auROC, "%.4f    "%x.auPRC, "%.4f   "%x.recall_at_50_fdr, "%.4f   "%x.recall_at_10_fdr 
            #if i > 3: break
        print
        print
    return

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

        #JIN
        # if not sample == 'FOXA2': continue
        # if not tf_name == 'liver': continue

        # print tf_name, sample
        print(sample, tf_name)
        # print bootstrapped_submissions[0]
        # user_ranks = defaultdict(list)
        # for index, submissions in bootstrapped_submissions.iteritems():
        #     # JIN: removed
        #     # print "\tbefore:",len(submissions)
        #     submissions = filter_older_submissions(submissions)
        #     # print "\tafter:",len(submissions)
        #     ranks = calc_combined_ranks(submissions)
        #     obs_users = set(x[0] for x in ranks.keys())
        #     for (principalId, submission_id), rank in ranks.iteritems():
        #         # print principalId, rank
        #         # user_ranks[(principalId, submission_id)].append(rank)
        #         user_ranks[(principalId, 0)].append(rank)
        #         global_scores[index][principalId].append(
        #             min(0.5, rank/(len(ranks)+1))
        #         )            
        #     for principalId in all_users - obs_users:
        #         global_scores[index][principalId].append(0.5)

        # #for (principalId, submission_id), ranks in sorted(
        # cnt = 1
        # for (principalId, _), ranks in sorted(
        #         user_ranks.iteritems(), key=lambda x: sorted(x[1])[1]):
        #     # print principalId, get_name(principalId), sorted(ranks)[1]
        #     # print principalId, sorted(ranks)[1]
        #     # print principalId, get_name(principalId), cnt, sorted(ranks)[1]
        #     # print principalId, cnt, sorted(ranks)[1]
        #     print principalId, get_name(principalId), sorted(ranks)[1], sorted(ranks)
        #     cnt += 1
        # print

        user_ranks = defaultdict(list)
        for index, submissions in bootstrapped_submissions.iteritems():
            # JIN: removed
            # print "\tbefore:",len(submissions)
            submissions = filter_older_submissions(submissions)
            # print "\tafter:",len(submissions)
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
        cnt = 1
        for (principalId, _), ranks in sorted(
                user_ranks.iteritems(), key=lambda x: sum(x[1])/len(x[1])):
            print(principalId) #, get_name(principalId), cnt #, sum(ranks)/len(ranks), sorted(ranks)
            cnt += 1
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
    print("\t".join(
        ("name".ljust(50), "rank", "lb", "mean", "ub")))
    for x in global_data: 
        print("%s\t%.2f\t%.2f\t%.2f\t%.2f" % (
            x.name.ljust(50), x.rank, x.score_lb, x.score_mean, x.score_ub))
    return rv, global_data

def calculate_ranks_from_DB_final_1_and_2(DB, ids_to_keep=None):

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
        #JIN
        # if not sample == 'FOXA2': continue
        # if not tf_name == 'liver': continue
        # print tf_name, sample
        print("# %s %s Conference/Final Round" % (sample, tf_name))
        print("Syanpse ID | Team name | Rank")
        print("---|---|---")
        user_ranks = defaultdict(list)
        for index, submissions in bootstrapped_submissions.iteritems():
            # JIN: removed
            # print "\tbefore:",len(submissions)
            # submissions = filter_older_submissions(submissions)
            # print "\tafter:",len(submissions)

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
    print("\t".join(
        ("name", "rank", "lb", "mean", "ub")))
    for x in global_data: 
        print("%s\t%.2f\t%.2f\t%.2f\t%.2f" % (
            x.name, x.rank, x.score_lb, x.score_mean, x.score_ub))

    print("# Conference/Final Round Overall Results")
    print(" | ".join()
        ("Team name", "rank", "Lower bound", "Mean", "Upperbound"))
    print("|".join(("---",)*6))
    for x in global_data: 
        print("%s | %.2f | %.2f | %.2f | %.2f" % (
            x.name, x.rank, x.score_lb, x.score_mean, x.score_ub))

    return rv, global_data

def update_global_scores_table(global_data):
    import challenge_config as config
    from synapseclient import Schema, Column, Table, Row, RowSet, as_table_columns
    # 'principalId', 'name', 'score_lb', 'score_mean', 'score_ub', 'rank'
    cols = [
        Column(name='UserID', columnType='STRING', maximumSize=100),
        Column(name='Name', columnType='STRING', maximumSize=100),
        Column(name='score_lb', columnType='DOUBLE'),
        Column(name='score_mean', columnType='DOUBLE'),
        Column(name='score_ub', columnType='DOUBLE'),
        Column(name='rank', columnType='DOUBLE'),
    ]
    schema = Schema(name='Global Scores Round 2', columns=cols, parent=config.CHALLENGE_SYN_ID)
    print(schema)
    results = syn.tableQuery("select * from {}".format('syn7416675'))
    if len(results) > 0:
        a = syn.delete(results.asRowSet())
    table = syn.store(Table(schema, global_data))
    results = syn.tableQuery("select * from {}".format(table.tableId))
    for row in results:
        print(row)
    return

def update_ranks(evaluation, dry_run=False):
    if type(evaluation) != Evaluation:
        evaluation = syn.getEvaluation(evaluation)

    ranks, global_data = calculate_ranks_from_DB(DB)
    #print sorted(ranks.keys())
    #previous_submissions = conf.collect_previous_submissions(syn, evaluation)

    for submission, status in syn.getSubmissionBundles(evaluation, status='SCORED'):
        submission_id = int(submission['id'])
        current_annotations = synapseclient.annotations.from_submission_status_annotations(
            status["annotations"])
        rank = ranks[submission_id] if submission_id in ranks else float('NaN')
        print(submission_id, rank)
        current_annotations['rank'] = rank
        status.annotations = synapseclient.annotations.to_submission_status_annotations(
            current_annotations, is_private=False)
        status = syn.store(status)

    # update the global data table
    update_global_scores_table(global_data)


def calc_bootstrapped_scores(labels, scores):
    from sklearn.cross_validation import StratifiedKFold
    for i, (indices, _) in enumerate(
            StratifiedKFold(labels, n_folds=10, random_state=0)):
        results = ClassificationResult(
            labels[indices], scores[indices].round(), scores[indices])
        yield i, results
    return

if __name__ == '__main__':
    #estimate_bootstrapped_scores_from_final_submissions_dir(
    #     DB,
    #     SUBMISSIONS_DIR,
    #     score_final_main)
    #SUBMISSION_QUEUE_ID = 7373880
    #estimate_bootstrapped_scores_from_submission_queue(
    #    DB, SUBMISSION_QUEUE_ID, score_final_main)

    # calculate_ranks_from_DB(DB)    
    calculate_ranks_from_DB_final_1_and_2(DB)

    # build_ranks_spreadsheet("/srv/scratch/nboley/encode/final/scores.db")
    #build_ranks_spreadsheet("/encode/final2/bootrapped_scores.db")
    # build_ranks_spreadsheet("/srv/scratch/leepc12/DREAM_challenge/final2/bootstrapped_scores.db")
    #build_ranks_spreadsheet(DB)
    #update_ranks(SUBMISSION_QUEUE_ID)
    #build_global_scores_table()
