#!/bin/bash -v

STAGE=$1
SRC=$2
TRG=$3
FUNCTION=$4
NSEEDS=$5

echo Running experiments with the following parameters:
echo STAGE: $STAGE
echo SRC: $SRC
echo TRG: $TRG
echo FUNCTION: $FUNCTION
echo NSEEDS: $NSEEDS
echo CUDA_VISIBLE_DEVICES is $CUDA_VISIBLE_DEVICES

###############################################################################
# Validate Input.

# Validate SRC/TRG language are present.
if [ -z "$SRC" ] || [ -z "$TRG" ]; then
	echo Must specify source and target languages. Exiting. && exit
fi
AVAILABLE_FUNCTIONS=(proc sgm)
if [[ ! "${AVAILABLE_FUNCTIONS[@]}" =~ "${FUNCTION}" ]]; then
	echo Must choose proc or sgm for function. Exiting. && exit
fi
AVAILABLE_STAGES=(single add-all active-learn stoch-add)
if [[ ! "${AVAILABLE_STAGES[@]}" =~ "${STAGE}" ]]; then
	echo Must choose add-all, active-learn, or stoch-add. Exiting. && exit
fi

###############################################################################
# Set variables.

MAX_EMBS=200000
MAX_SEEDS=4800
STOCH_ADD_INTERVAL=100

PAIRS=`pwd`/dicts/$SRC-$TRG/train/$SRC-$TRG.0-5000.txt.1to1
SRC_EMBS=`pwd`/embs/wiki.$SRC.vec
TRG_EMBS=`pwd`/embs/wiki.$TRG.vec
NEW_NSEEDS_PER_ROUND=-1
MIN_PROB=0.0
NORM=(unit center unit)
ACTIVE_LEARNING=

################################################################################

if [ $STAGE == 'single' ]; then
	ITERS=1
elif [ $STAGE == 'add-all' ]; then
	ITERS=10
elif [ $STAGE == 'active-learn' ]; then
	ACTIVE_LEARNING=--active-learning
	ITERS=10
elif [ $STAGE == 'stoch-add' ]; then
	NEW_NSEEDS_PER_ROUND=(`seq $NSEEDS $STOCH_ADD_INTERVAL $MAX_SEEDS`)
	ITERS=${#NEW_NSEEDS_PER_ROUND[@]}
fi

################################################################################

OUTDIR=`pwd`/exps/$SRC-$TRG/$FUNCTION/$STAGE/$NSEEDS
mkdir -p $OUTDIR
python proc_v_sgm.py --src-embs $SRC_EMBS --trg-embs $TRG_EMBS \
	--norm ${NORM[@]} --function $FUNCTION --pairs $PAIRS --n-seeds $NSEEDS \
	--max-embs $MAX_EMBS --min-prob $MIN_PROB --proc-iters $ITERS \
	--softsgm-iters 1 --diff-seeds-for-rev \
	--iterative-softsgm-iters $ITERS $ACTIVE_LEARNING \
	--new-nseeds-per-round ${NEW_NSEEDS_PER_ROUND[@]} > $OUTDIR/run.out
