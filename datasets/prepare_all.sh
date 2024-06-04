#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Preparing gsm8k"
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name train
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name test
# train + validation - should be used for the final run only
python3 ${SCRIPT_DIR}/gsm8k/prepare.py --split_name train_full
echo "Preparing gsm-plus"
python3 ${SCRIPT_DIR}/gsm-plus/prepare.py     # test set
echo "Preparing gsm-ic-mstep"
python3 ${SCRIPT_DIR}/gsm-ic-mstep/prepare.py # test set
echo "Preparing gsm-ic-2step"
python3 ${SCRIPT_DIR}/gsm-ic-2step/prepare.py # test set
echo "Preparing gsm-hard"
python3 ${SCRIPT_DIR}/gsm-hard/prepare.py     # test set
echo "Preparing tabmwp"
python3 ${SCRIPT_DIR}/tabmwp/prepare.py       # test set
echo "Preparing mawps"
python3 ${SCRIPT_DIR}/mawps/prepare.py        # test set
echo "Preparing asdiv"
python3 ${SCRIPT_DIR}/asdiv/prepare.py        # test set
echo "Preparing algebra222"
python3 ${SCRIPT_DIR}/algebra222/prepare.py   # test set
echo "Preparing svamp"
python3 ${SCRIPT_DIR}/svamp/prepare.py        # test set
echo "Preparing functional"
python3 ${SCRIPT_DIR}/functional/prepare.py   # test set
echo "Preparing math"
python3 ${SCRIPT_DIR}/math/prepare.py --split_name train
python3 ${SCRIPT_DIR}/math/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/math/prepare.py --split_name test
python3 ${SCRIPT_DIR}/math/prepare.py --split_name train_full

# prepare datasets with masked solutions
echo "Preparing math-masked"
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name train
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/math-masked/prepare.py --split_name train_full
echo "Preparing gsm8k-masked"
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name train
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name validation
python3 ${SCRIPT_DIR}/gsm8k-masked/prepare.py --split_name train_full
