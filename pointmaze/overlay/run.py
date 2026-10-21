from search.configs import Arguments
from search.script_utils import get_pipe, get_args

def main(dataset: str="pointmaze-giant-navigate-v0", method: str='dfs', device: str="cuda",
         sampler: str='vanilla', lam_start: float=1.0, lam_end: float=1.0, n_replicas: int=4, seed: int=42):
    args = Arguments()
    args.device = device
    args.dataset = dataset
    args.method = method
    args.sampler = sampler
    args.lam_start = lam_start
    args.lam_end = lam_end
    args.n_replicas = n_replicas
    args.seed = seed
    args_grid = get_args(args)

    for args in args_grid:
        pipe = get_pipe(args)
        returns = pipe.experiment()
        success_rate = returns['average']['total_reward']
        average_compute = returns['average']['compute']
        print(f"Success Rate: {success_rate}, Average Compute: {average_compute}")
        output_file = f'results_{args.method}_{args.dataset}.txt'
        with open(output_file, 'a') as f:
            f.write(f"Maze: {args.dataset} | Sampler: {args.sampler} | Lam {args.lam_start} to {args.lam_end} "
                    f"with {args.n_replicas} | Method: {args.method} | Particles: {args.per_sample_batch_size} | "
                    f"Seed: {args.seed} | Compute: {average_compute} | Success Rate: {success_rate}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference scaling experiment.")
    parser.add_argument('--dataset',
                        type=str,
                        default='pointmaze-ultra-navigate-v0',
                        choices=['pointmaze-giant-navigate-v0', 'pointmaze-ultra-navigate-v0'],
                        help='Maze env to use for the experiment.')
    parser.add_argument('--method',
                        type=str,
                        default='bon',
                        choices=['dfs', 'bon', 'bfs-resampling', 'bfs-pruning'],
                        help='Search method to use')
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        )
    parser.add_argument('--sampler',
                        type=str,
                        default='vanilla',
                        choices=['vanilla', 'tsr', 'steer', 'cns', 'mcmc', 'crepe'],
                        help='Sampler run inside the search method')
    parser.add_argument('--lam_start', type=float, default=1.0, help='First rung of the lambda ladder; also the scale used by cns and mcmc')
    parser.add_argument('--lam_end', type=float, default=1.0, help='Last rung of the lambda ladder')
    parser.add_argument('--n_replicas', type=int, default=4, help='Number of rungs; also the particle count for bon')
    parser.add_argument('--seed', type=int, default=42)
    cli_args = parser.parse_args()

    main(dataset=cli_args.dataset, method=cli_args.method, device=cli_args.device, sampler=cli_args.sampler,
         lam_start=cli_args.lam_start, lam_end=cli_args.lam_end, n_replicas=cli_args.n_replicas, seed=cli_args.seed)
