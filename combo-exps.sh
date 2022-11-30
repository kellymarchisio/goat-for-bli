#!/bin/bash -v

SRC=$1
TRG=$2
START_FUNCTION=$3
NSEEDS=$4
INIT=$5
GM_FUNCTION=$6 # Graph-Matching Function (SGM vs. GOAT)

echo Running experiments with the following parameters:
echo SRC: $SRC
echo TRG: $TRG
echo START_FUNCTION: $START_FUNCTION
echo NSEEDS: $NSEEDS
echo INIT: $INIT
echo GM_FUNCTION: $GM_FUNCTION
echo CUDA_VISIBLE_DEVICES is $CUDA_VISIBLE_DEVICES

###############################################################################
# Validate Input.

# Validate SRC/TRG language are present.
if [ -z "$SRC" ] || [ -z "$TRG" ]; then
	echo Must specify source and target languages. Exiting. && exit
fi
AVAILABLE_START_FUNCTIONS=(proc sgm goat)
if [[ ! "${AVAILABLE_START_FUNCTIONS[@]}" =~ "${START_FUNCTION}" ]]; then
	echo Must choose proc or sgm for function. Exiting. && exit
fi
AVAILABLE_GM_FUNCTIONS=(sgm goat)
if [ -z $GM_FUNCTION ]; then
	echo No Graph-Matching function set. Setting it to SGM
	GM_FUNCTION=sgm
fi
if [[ ! " ${AVAILABLE_GM_FUNCTIONS[@]} " =~ " ${GM_FUNCTION} " ]]; then
	echo Invalid Graph=Matching function. Exiting. && exit
fi

# Validate SGM/GOAT initialization type (randomized or barycenter).
AVAILABLE_INITS=(randomized barycenter)
if [ -z $INIT ]; then
       echo No Initialization provided. Setting it to \'randomized\'
       echo \t note: this only applies to graph-matching algorithms
       INIT=randomized
fi
if [[ ! " ${AVAILABLE_INITS[@]} " =~ " ${INIT} " ]]; then
	echo Invalid initialization. Exiting. && exit
fi

###############################################################################
# Set variables.

MAX_EMBS=200000
PAIRS=`pwd`/dicts/$SRC-$TRG/train/$SRC-$TRG.0-5000.txt.1to1
SRC_EMBS=`pwd`/embs/wiki.$SRC.vec
TRG_EMBS=`pwd`/embs/wiki.$TRG.vec
NEW_NSEEDS_PER_ROUND=-1
MIN_PROB=0.0
NORM=(unit center unit)
ACTIVE_LEARNING=
K=1
ITERATIVE_SGM_ITERS=1
PROC_ITERS=5
SOFTSGM_ITERS=1

###############################################################################

OUTDIR=`pwd`/exps/$SRC-$TRG/combo/start-$START_FUNCTION/gm-$GM_FUNCTION-$INIT/$NSEEDS
mkdir -p $OUTDIR
python combo.py --src-embs $SRC_EMBS --trg-embs $TRG_EMBS \
	--norm ${NORM[@]} --start $START_FUNCTION --pairs $PAIRS --n-seeds $NSEEDS \
	--max-embs $MAX_EMBS --min-prob $MIN_PROB --proc-iters $PROC_ITERS \
	--softsgm-iters $SOFTSGM_ITERS --k 1 $DIFF_SEEDS_FOR_REV \
	--iterative-softsgm-iters $ITERATIVE_SGM_ITERS \
	--new-nseeds-per-round ${NEW_NSEEDS_PER_ROUND[@]} \
	--init $INIT --function $GM_FUNCTION > $OUTDIR/run.out
