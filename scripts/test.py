from dos2bandnet.train_core import evaluate_ldm_ckpt_metrics, view_ranked_from_ldm_ckpt

out_root = "/home/yrjin/band-prediction/mp-dataset/runs"
# run_name="exp-jvcumgq8"
# run_name="exp-kzbt180m"
# run_name="exp-kzbt180m/finetune1"
run_name="exp-jvcumgq8/finetune14"

metrics = evaluate_ldm_ckpt_metrics(
    out_root=out_root,
    run_name=run_name,
    split="test",
    steps=50,
    guidance=1.0,
)
print(metrics)

ret = view_ranked_from_ldm_ckpt(
    out_root=out_root,
    run_name=run_name,
    split="test",
    steps=50,
    guidance=1.0,
    save_png=False,
)
print(ret["error_csv"])