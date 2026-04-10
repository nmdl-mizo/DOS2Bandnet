#!/usr/bin/env python3
import multiprocessing as mp
from dos2bandnet.wandb_runner import (
    create_sweep, run_agent, run_agent_pool, replay_from_run
)

def parse_args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="create",
                    choices=["create", "agent", "agent_pool", "replay"])
    ap.add_argument("--project", type=str, default="band-prediction")
    ap.add_argument("--entity", type=str, default="yeongrokj95-pnu")
    ap.add_argument("--out_root", type=str, default="runs")
    ap.add_argument("--run_name_prefix", type=str, default="exp")
    ap.add_argument("--prefix1", type=str, default="mp-")
    ap.add_argument("--prefix2", type=str, default="ebs_")
    ap.add_argument("--sweep_id", type=str, default="")
    ap.add_argument("--gpu", type=str, default=None)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--pool_gpus", type=str, default="")
    ap.add_argument("--agents_per_gpu", type=int, default=1)
    ap.add_argument("--count_per_agent", type=int, default=0)
    ap.add_argument("--from_run", type=str, default="")
    ap.add_argument("--lr_diff_new", type=float, default=None)
    ap.add_argument("--override", type=str, default="")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.mode == "create":
        create_sweep(args)
        return

    if args.mode == "agent":
        run_agent(args.sweep_id, count=args.count, args=args)
        return

    if args.mode == "agent_pool":
        gpus = [int(x) for x in args.pool_gpus.split(",") if x.strip()]
        run_agent_pool(
            sweep_id=args.sweep_id,
            pool_gpus=gpus,
            agents_per_gpu=args.agents_per_gpu,
            count_per_agent=int(args.count_per_agent or args.count),
            args=args,
        )
        return

    if args.mode == "replay":
        replay_from_run(args)
        return

if __name__ == "__main__":
    # if it needs:
    # mp.set_start_method("spawn", force=True)
    main()
