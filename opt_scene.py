import argparse
import os

from tqdm.auto import trange

import drjit as dr
import mitsuba as mi

import gotex.logger as logger
from gotex.trainer import Trainer
from gotex.config import ExperimentConfig, create_runtime, load_config

def main(args, extra):
    config: ExperimentConfig = load_config(args.config, cli_args=extra)
    runtime = create_runtime(seed=config.seed, device=config.device)

    trainer = Trainer(
        config=config.trainer,
        runtime=runtime,
    )

    out_folder = f'{config.exp_root_dir}/{config.name}'
    os.makedirs(out_folder, exist_ok=True)
    logger.set_out_dir(out_folder)

    iterator = trange(trainer.cfg.max_steps, desc="Optimizing", disable=not config.use_tqdm)
    for step_idx in iterator:
        
        image, loss = trainer.step()            

        if step_idx % trainer.cfg.save_every == 0:
            if config.use_tqdm:
                iterator.set_postfix(loss=loss.item())
            else:
                dr.print("step {step_idx} | {loss=}", step_idx=step_idx, loss=loss)
            

            logger.save_image(image, f'render_{step_idx}.png')
            for k, v in trainer.opt.items():
                logger.save_image(v, f'{k.split(".")[0]}.exr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize scene texture parameters with Stable Diffusion guidance.")
    parser.add_argument("--config", required=True, help="path to config file")
    args, extra = parser.parse_known_args()

    main(args, extra)
