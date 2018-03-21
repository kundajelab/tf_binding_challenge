##-----------------------------------------------------------------------------
##
## challenge specific code and configuration for Encode Challenge
##
##-----------------------------------------------------------------------------
import os
from score import *
from synapseclient.annotations import from_submission_status_annotations
from collections import defaultdict

## A Synapse project will hold the assetts for your challenge. Put its
## synapse ID here, for example
## CHALLENGE_SYN_ID = "syn1234567"
CHALLENGE_SYN_ID = "syn6131484"

## Name of your challenge, defaults to the name of the challenge's project
CHALLENGE_NAME = "ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge"

SCORED_MESSAGE = """\
Thanks for your submission to the ENCODE-DREAM Challenge!
  round: {round}
  transcription factor: {tf}
  cell-line: {cell_line}
"""

## Configure the location where submissions will be downloaded
## None means submissions will be stored in the Synapse Cache (~/.synapseCache/)
SUBMISSION_DIR = '/encode/submissions'

## Synapse user IDs of the challenge admins who will be notified by email
## about errors in the scoring script
ADMIN_USER_IDS = [1421212, 3341836] # chris.bare, nboley

## Each question in your challenge should have an evaluation queue through
## which participants can submit their predictions or models. The queues
## should specify the challenge project as their content source. Queues
## can be created like so:
##   evaluation = syn.store(Evaluation(
##     name="My Challenge Q1",
##     description="Predict all the things!",
##     contentSource="syn1234567"))
## ...and found like this:
##   evaluations = list(syn.getEvaluationByContentSource('syn3375314'))
## Configuring them here as a list will save a round-trip to the server
## every time the script starts.

evaluation_queues = [
    {'contentSource': 'syn6131484',
     'createdOn': '2016-07-15T23:10:24.111Z',
     'description': 'TEST submission queue for the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge',
     'etag': '0edeac65-1835-4739-aa00-015c6734740c',
     'id': '7065092',
     'name': 'ENCODE TF Challenge TEST',
     'ownerId': '3343430',
     'status': 'OPEN'},
    {'contentSource': 'syn6131484',
     'createdOn': '2016-07-21T22:58:32.334Z',
     'description': 'Submission queue for the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge',
     'etag': '20569083-abdb-4b36-933e-7f31e68580ce',
     'id': '7071644',
     'name': 'ENCODE-DREAM Transcription Factor Binding Site Prediction Challenge',
     'ownerId': '3343430',
     'status': 'OPEN',
     'submissionInstructionMessage': 'For information on how to make submissions to the challenge please refer to: https://www.synapse.org/#!Synapse:syn6131484/wiki/402044.\n',
     'submissionReceiptMessage': 'Thank you for participating in the ENCODE-DREAM in vivo Transcription Factor Binding Site Prediction Challenge. Your submission has been queued for scoring.\n'}]
evaluation_queue_by_id = {q['id']:q for q in evaluation_queues}

## define the default set of columns that will make up the leaderboard
LEADERBOARD_COLUMNS = [
    dict(name='objectId',      display_name='ID',      columnType='STRING', maximumSize=20),
    dict(name='userId',        display_name='User',    columnType='STRING', maximumSize=20, renderer='userid'),
    dict(name='entityId',      display_name='Entity',  columnType='STRING', maximumSize=20, renderer='synapseid'),
    dict(name='versionNumber', display_name='Version', columnType='INTEGER'),
    dict(name='name',          display_name='Name',    columnType='STRING', maximumSize=240),
    dict(name='team',          display_name='Team',    columnType='STRING', maximumSize=240)]

## Here we're adding columns for the output of our scoring functions, score,
## rmse and auc to the basic leaderboard information. In general, different
## questions would typically have different scoring metrics.
leaderboard_columns = {}
for q in evaluation_queues:
    leaderboard_columns[q['id']] = LEADERBOARD_COLUMNS + [
        dict(name='auROC',            display_name='auROC',           columnType='DOUBLE'),
        dict(name='auPRC',            display_name='auPRC',           columnType='DOUBLE'),
        dict(name='recall_at_05_fdr', display_name='Recall @ 5% FDR', columnType='DOUBLE'),
        dict(name='recall_at_10_fdr', display_name='Recall @10% FDR', columnType='DOUBLE'),
        dict(name='recall_at_25_fdr', display_name='Recall @25% FDR', columnType='DOUBLE')
    ]


## list of transcription factors for the leaderboard round
tf_cell_line_pairs = [
    ('ARID3A','K562'),
    ('ATF2',  'K562'),
    ('ATF3',  'liver'),
    ('ATF7',  'MCF-7'),
    ('CEBPB', 'MCF-7'),
    ('CREB1', 'MCF-7'),
    ('CTCF',  'GM12878'),
    ('E2F6',  'K562'),
    ('EGR1',  'K562'),
    ('EP300', 'MCF-7'),
    ('FOXA1', 'MCF-7'),
    ('GABPA', 'K562'),
    ('GATA3', 'MCF-7'),
    ('JUND',  'H1-hESC'),
    ('MAFK',  'K562'),
    ('MAFK',  'MCF-7'),
    ('MAX',   'MCF-7'),
    ('MYC',   'HepG2'),
    ('REST',  'K562'),
    ('RFX5',  'HepG2'),
    ('SPI1',  'K562'),
    ('SRF',   'MCF-7'),
    ('STAT3', 'GM12878'),
    ('TAF1',  'HepG2'),
    ('TCF12', 'K562'),
    ('TCF7L2','MCF-7'),
    ('TEAD4', 'MCF-7'),
    ('YY1',   'K562'),
    ('ZNF143','K562')]

tf_cell_line_pairs_final = [
    ('CTCF',  'PC-3'),
    ('CTCF',  'induced_pluripotent_stem_cell'),
    ('E2F1',  'K562'),
    ('EGR1',  'liver'),
    ('FOXA1', 'liver'),
    ('FOXA2', 'liver'),
    ('GABPA', 'liver'),
    ('HNF4A', 'liver'),
    ('JUND',  'liver'),
    ('MAX',   'liver'),
    ('NANOG', 'induced_pluripotent_stem_cell'),
    ('REST',  'liver'),
    ('TAF1',  'liver'),
]

## Submission limits
DEFAULT_LIMIT = 1000
LIMITS = {(tf,cl):10 for tf,cl in tf_cell_line_pairs}
LIMITS_final = {(tf,cl):10 for tf,cl in tf_cell_line_pairs_final}


def collect_previous_submissions(syn, evaluation):
    """
    Count submissions to enforce submission limits
    """
    results = defaultdict(set)
    for submission, status in syn.getSubmissionBundles(evaluation, status='SCORED'):
        principalId = submission.teamId if 'teamId' in submission else submission.userId
        annotations = from_submission_status_annotations(status.annotations)

        if annotations['round'] == 'L':
            results[(principalId, annotations['tf'], annotations['cell_line'])].add(submission.id)

    return results


def parse_filename(filename):
    """
    Parse filenames of the form F.CTCF.HepG2.tab.gz or
    my.method.name.{L,F,B}.[TF].[cell-line](?:\(\d+\))?.tab.gz
    """
    m = re.match(fname_pattern, filename)
    if m:
        return (m.group(1), m.group(2), m.group(3))
    else:
        raise InputError("The submitted filename ({}) does not match expected naming pattern '{}'".format(
            filename, fname_pattern))


def score_submission(evaluation, submission, previous_submissions):
    """
    Find the right scoring function and score the submission

    :param evaluation: a synapse evaluation queue object
    :param submission: a synapse submission object
    :param previous_submissions: a mapping from principalID (team or user),  to
        a list of ValidationResults objects. See collect_previous_submissions().

    :returns: (score, message) where score is a dict of stats and message
              is text for display to user
    """
    filename = os.path.basename(submission.filePath)
    challenge_round, tf, cell_line = parse_filename(filename)
    principalId = submission.teamId if 'teamId' in submission else submission.userId

    ## rename the submission locally by prepending principal ID and submission ID
    new_filename = "{}.{}.{}".format(principalId, submission.id, os.path.basename(submission.filePath))
    new_filepath = os.path.join(os.path.dirname(submission.filePath), new_filename)
    os.rename(submission.filePath, new_filepath)

    ## Enforce submission limits on leaderboard round
    if challenge_round == 'L':

        ## check for a valid tf/cell-line combination according to
        ##   https://www.synapse.org/#!Synapse:syn6044646/wiki/401935
        if (tf, cell_line) not in LIMITS:
            raise InputError("The submitted factor, sample combination ({}, {}) is not valid.".format(
                             tf, cell_line))

        ## check whether we're over the quota
        num_submissions = len(previous_submissions[(principalId, tf, cell_line)]) + 1
        limit = LIMITS[(tf, cell_line)]
        print "Submission limits: {} made submission {} for the tf/cell-line {}/{} out of allowed {}.".format(principalId, num_submissions, tf, cell_line, limit)
        if num_submissions > limit:
            raise InputError("You've submitted {} predictions for the tf/cell-line combination {}/{} but the limit is {}. Your submission won't be scored.".format(
                             num_submissions, tf, cell_line, limit))
    elif challenge_round == 'F':
        if (tf, cell_line) not in LIMITS_final:
            raise InputError("The submitted factor, sample combination ({}, {}) is not valid.".format(
                             tf, cell_line))
    result = score_main(new_filepath)
    # JIN
    assert(False)
    # modify fake score file!!!!!!!!!

    ## returns a ClassificationResult
    stats = {field:getattr(result, field) for field in ClassificationResultData._fields}

    ## forward housekeeping info along
    stats['tf'] = tf
    stats['cell_line'] = cell_line
    stats['round'] = challenge_round

    ## update the counts if we successfully scored the submission
    if challenge_round == 'L':
        previous_submissions[(principalId, tf, cell_line)].add(submission.id)

    return (stats, SCORED_MESSAGE.format(tf=tf, cell_line=cell_line, round=challenge_round))
