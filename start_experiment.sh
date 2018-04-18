#!/usr/bin/env bash
if [ -z "$LOGDIR" ]; then
    echo "error: provide a value for LOGDIR variable"
    exit 1
else
    echo "using logdir=$LOGDIR"
fi
if [ -z "$CONFIG_FILE" ]; then
    echo "error: provide a value for CONFIG_FILE variable"
    exit 1
else
    echo "using config_file=$CONFIG_FILE";
fi
if [ -z "$MODE" ]; then
    echo "error: provide a value for MODE variable"
    exit 1
else
    echo "using mode=$MODE"
fi
if [ -z "$CONTINUE_LEARNING" ]; then
    echo "error: provide a value for CONTINUE_LEARNING variable"
    exit 1
else
    echo "using continue_learning=$CONTINUE_LEARNING"
fi
if [ $CONTINUE_LEARNING == 1 ]; then
    if [ ! -d "$LOGDIR" ]; then
        echo "error: $LOGDIR does not exist, can't continue learning"
        exit 1
    fi
else
    if [ -d "$LOGDIR" ]; then
        echo "error: $LOGDIR already exist, use CONTINUE_LEARNING=1 if you want to continue learning"
        exit 1
    fi
fi

mkdir -p ${LOGDIR}

LOGFILE=${LOGDIR}/output_$(date +%Y%m%d_%H%M).log
echo "Logging the output to ${LOGFILE}"

GITFILE=${LOGDIR}/gitinfo_$(date +%Y%m%d_%H%M).log
echo "Logging git information to ${GITFILE}"

echo "commit hash: $(git rev-parse HEAD)" > ${GITFILE}
echo -e "\n$(git diff)" >> ${GITFILE}

cp ${CONFIG_FILE} ${LOGDIR}/config_$(date +%Y%m%d_%H%M).py

if [ $CONTINUE_LEARNING == 1 ]; then
    python -u run.py --config_file=${CONFIG_FILE} \
                     --mode=${MODE} \
                     --continue_learning \
                     --logdir=${LOGDIR}/logs \
                     2>&1 | tee ${LOGFILE}
else
    python -u run.py --config_file=${CONFIG_FILE} \
                     --mode=${MODE} \
                     --logdir=${LOGDIR}/logs \
                     --no_dir_check \
                     2>&1 | tee ${LOGFILE}
fi

