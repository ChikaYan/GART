dataset="xhuman" # in ins-ava processing formart
profile="ubc_mlp"
logbase=${profile}

SEQUENCES=(
    "00019"
)

for seq in ${SEQUENCES[@]}; do
    python solver.py --profile ./profiles/ubc/${profile}.yaml --dataset $dataset --seq ${seq} --logbase logs/$logbase --fast

    # python solver.py --profile ./profiles/ubc/${profile}.yaml \
    #     --dataset $dataset --seq ${seq} --eval_only --log_dir $logbase

done