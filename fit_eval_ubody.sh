
# remeber to search for "UBODY_DATA_PATH" and change it to your path (in total three occurrences)

dataset="ubody" # in ins-ava processing formart
profile="ubc_mlp"
logbase=${profile}

SEQUENCES=(
    "001"
)

for seq in ${SEQUENCES[@]}; do
    python solver.py --profile ./profiles/ubc/${profile}.yaml --dataset $dataset --seq ${seq} --logbase logs/$logbase --fast

    # python solver.py --profile ./profiles/ubc/${profile}.yaml \
    #     --dataset $dataset --seq ${seq} --eval_only --log_dir $logbase

done