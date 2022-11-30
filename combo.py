###############################################################################
#
# Iterative Procrustes + SGM & GOAT Combination System.
#
# Written by Kelly Marchisio, 2020-2021.
#
###############################################################################

import argparse
from third_party.vecmap import embeddings
import proc_v_sgm

def main(args):
    # Process data. Get train/dev split, seeds.
    (word_pairs, src_embs, src_word2ind, src_ind2word, trg_embs, trg_word2ind,
            trg_ind2word, oov_word_pairs) = proc_v_sgm.load_embs_and_wordpairs(args)
    print('OOV Word Pairs:', oov_word_pairs)
    _, (train_inds, dev_inds) = proc_v_sgm.create_train_dev_split(word_pairs,
            args.n_seeds, src_word2ind, trg_word2ind, args.randomize_seeds)
    gold_src_train_inds, gold_trg_train_inds = proc_v_sgm.unzip_pairs(train_inds)
    src_dev_inds, trg_dev_inds = proc_v_sgm.unzip_pairs(dev_inds)

    # Normalize embeddings in-place.
    print('Normalizing embeddings...')
    embeddings.normalize(src_embs, args.norm)
    embeddings.normalize(trg_embs, args.norm)
    print('Done normalizing embeddings.')
    # Make similarity matrices.

    xxT = src_embs @ src_embs.T
    yyT = trg_embs @ trg_embs.T

    sgm_hyps_src = []
    sgm_hyps_trg = []
    gm_opts = dict(shuffle_input=True, maximize=True, P0=args.init)

    for i in range(10):
        print('----------------------------------')
        print('----------------------------------')
        print('Starting Iteration', i)
        print('----------------------------------')

        if args.start == 'proc':
            print('\nRunning Iterative Procrustes for {0} iterations'.format(
                args.proc_iters), flush=True)
            _, _, proc_hyps_int, _, _ = proc_v_sgm.iterative_procrustes_w_csls(src_embs, trg_embs,
                    sgm_hyps_src, sgm_hyps_trg, gold_src_train_inds,
                    gold_trg_train_inds, dev_inds, args.new_nseeds_per_round,
                    total_i=args.proc_iters,
                    diff_seeds_for_rev=args.diff_seeds_for_rev, k=args.k)
            print('Running Graph Matching:', args.function, flush=True)
            proc_hyps_src, proc_hyps_trg = proc_v_sgm.unzip_pairs(proc_hyps_int)
            hyps, _, sgm_hyps_int = proc_v_sgm.iterative_softsgm(xxT, yyT, proc_hyps_src,
                    proc_hyps_trg, gold_src_train_inds, gold_trg_train_inds,
                    args.softsgm_iters, args.k, args.min_prob, dev_inds,
                    args.new_nseeds_per_round, curr_i=1,
                    total_i=args.iterative_softsgm_iters,
                    diff_seeds_for_rev=args.diff_seeds_for_rev,
                    run_reverse=True, function=args.function, opts=gm_opts)
            sgm_hyps_src, sgm_hyps_trg = proc_v_sgm.unzip_pairs(sgm_hyps_int)

        elif args.start == 'sgm' or args.start == 'goat':
            print('Running Graph Matching:', args.function, flush=True)
            _, _, sgm_hyps_int = proc_v_sgm.iterative_softsgm(xxT, yyT,
                    sgm_hyps_src, sgm_hyps_trg,
                    gold_src_train_inds, gold_trg_train_inds,
                    args.softsgm_iters, args.k, args.min_prob, dev_inds,
                    args.new_nseeds_per_round, curr_i=1,
                    total_i=args.iterative_softsgm_iters,
                    diff_seeds_for_rev=args.diff_seeds_for_rev,
                    run_reverse=True,
                    function=args.function, opts=gm_opts)
            print('\nRunning Iterative Procrustes for {0} iterations'.format(
                args.proc_iters), flush=True)
            sgm_hyps_src, sgm_hyps_trg = proc_v_sgm.unzip_pairs(sgm_hyps_int)
            hyps, _, proc_hyps_int, _, _ = proc_v_sgm.iterative_procrustes_w_csls(src_embs, trg_embs,
                    sgm_hyps_src, sgm_hyps_trg, gold_src_train_inds,
                    gold_trg_train_inds, dev_inds, args.new_nseeds_per_round,
                    total_i=args.proc_iters,
                    diff_seeds_for_rev=args.diff_seeds_for_rev, k=args.k)
            sgm_hyps_src, sgm_hyps_trg = proc_v_sgm.unzip_pairs(proc_hyps_int)

    # Eval.
    dev_src_inds, dev_trg_inds = proc_v_sgm.unzip_pairs(dev_inds)
    dev_hyps = set(hyp for hyp in hyps if hyp[0] in dev_src_inds)
    matches, precision, recall = proc_v_sgm.eval(dev_hyps, dev_inds)
    print('\tDev Pairs matched: {0} \n\t(Precision; {1}%) (Recall: {2}%)'
            .format(len(matches), precision, recall), flush=True)



parser = argparse.ArgumentParser(description='LAP Experiments')
parser.add_argument('--src-embs', metavar='PATH', required=True,
    help='Path to source embeddings.')
parser.add_argument('--trg-embs', metavar='PATH', required=True,
    help='Path to target embeddings.')
parser.add_argument('--function', choices=['proc', 'sgm', 'goat'], required=True,
    help='Which function to run (Procrustes (proc), SGM (sgm), or GOAT ' +
    '(goat ).')
parser.add_argument('--init', choices=['randomized', 'barycenter'],
        default='randomized', help='P0 initialization for graph matching')
parser.add_argument('--start', choices=['proc', 'sgm', 'goat'], required=True,
        help='Whether to start with Iterative Procrustes, SGM, or GOAT.')
parser.add_argument('--norm', metavar='N', choices=['noop', 'unit', 'center'],
        nargs='+', required=True,
        help='How to normalize embeddings (can take multiple args)')
parser.add_argument('--max-embs', type=int, default=200000,
    help='Maximum num of word embeddings to use.')
parser.add_argument('--min-prob', type=float, default=0.0,
    help='The minimum probability to consider for softsgm')
parser.add_argument('--pairs', metavar='PATH', required=True,
    help='train seeds + dev pairs')
parser.add_argument('--n-seeds', type=int, required=True, help='Num train seeds to use')
parser.add_argument('--proc-iters', type=int, default=10,
    help='Rounds of iterative Procrustes to run.')
parser.add_argument('--iterative-softsgm-iters', type=int, default=1,
    help='Rounds of iterative SoftSGM to run.')
parser.add_argument('--softsgm-iters', type=int, default=1,
    help='Rounds of SoftSGM to run to create probdist.')
parser.add_argument('--k', type=int, default=1,
    help='How many hypotheses to return per source word.')
parser.add_argument('--randomize-seeds', action='store_true',
        help='If set, randomizes the seeds to use (instead of getting them in '
        'order from args.pairs file)')
parser.add_argument('--new-nseeds-per-round', metavar='N', type=int, nargs='+',
        default=-1, help='Number of seeds to add per round in iterative runs.')
parser.add_argument('--diff-seeds-for-rev', action='store_true',
    help='When running matching in reverse, regenerate seeds (if there are '
    'additional input seeds from a previous round, these will then be '
    'shuffled.')

args = parser.parse_args()

main(args)
