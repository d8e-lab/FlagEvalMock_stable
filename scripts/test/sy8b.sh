list=(
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-5-la32-ld0.1-e1-ml2048-wd0.01-2024-01-17_11-51-32
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r8-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_16-58-27
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r2-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_15-48-52
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr2e-4-la32-ld0.1-e1-ml2048-2024-01-17_14-38-19
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r32-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_19-10-47
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-5-la32-ld0.1-e2-ml2048-2024-01-17_12-16-57
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_18-01-20
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_18-07-26
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-4-la32-ld0.1-e1-ml2048-2024-01-17_13-20-49
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r2-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_15-41-07
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r32-lr1e-4-la32-ld0.4-e1-ml2048-2024-01-17_23-55-55
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-1e-5-32-0.1-2024-01-16_01-38-43
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-4-la32-ld0.2-e1-ml2048-2024-01-17_20-27-16
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-4-la32-ld0.1-e1-ml2048-2024-01-17_13-27-41
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r8-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_16-52-04
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr5e-5-la32-ld0.1-e1-ml2048-2024-01-17_12-16-47
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-4-la32-ld0.2-e1-ml2048-2024-01-17_22-39-21
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr2e-4-la32-ld0.1-e1-ml2048-2024-01-17_14-31-01
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r4-lr1e-4-la32-ld0.2-e1-ml2048-2024-01-17_20-20-41
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-4-la32-ld0.2-e1-ml2048-2024-01-17_22-46-27
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-4-la32-ld0.1-e1-ml2048-2024-01-17_21-30-09
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r16-lr1e-4-la32-ld0.1-e1-ml2048-2024-01-17_21-36-53
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r32-lr1e-5-la32-ld0.1-e1-ml2048-2024-01-17_19-17-07
/mnt/42_store/wyh/outputs/sy8b-longer-checkpoint-27000-data1211-hyper/r32-lr1e-4-la32-ld0.4-e1-ml2048-2024-01-17_23-48-41
)
# echo ${list[@]}
mkdir -p /mnt/SFT_store/3090_eval/FlagEvalMock_stable/evaluation_results/sy8b/
for model in "${list[@]}"; do
    # ls $model/merge
    bash wyh_sample/23_no_imdb.sh $model/merge 0,1,2,3,4,5 /mnt/SFT_store/3090_eval/FlagEvalMock_stable/evaluation_results/sy8b/
done