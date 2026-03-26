# File strongly inspired by the threestudio codebase

import os, random
from dataclasses import dataclass, field

from omegaconf import OmegaConf, DictConfig

import torch
import drjit as dr
import mitsuba as mi

from typing import Any, Optional, Union

# ============ Register OmegaConf Recolvers ============= #
OmegaConf.register_new_resolver(
    "calc_exp_lr_decay_rate", lambda factor, n: factor ** (1.0 / n)
)
OmegaConf.register_new_resolver("add", lambda a, b: a + b)
OmegaConf.register_new_resolver("sub", lambda a, b: a - b)
OmegaConf.register_new_resolver("mul", lambda a, b: a * b)
OmegaConf.register_new_resolver("div", lambda a, b: a / b)
OmegaConf.register_new_resolver("idiv", lambda a, b: a // b)
OmegaConf.register_new_resolver("basename", lambda p: os.path.basename(p))
OmegaConf.register_new_resolver("rmspace", lambda s, sub: s.replace(" ", sub))
OmegaConf.register_new_resolver("tuple2", lambda s: [float(s), float(s)])
OmegaConf.register_new_resolver("gt0", lambda s: s > 0)
OmegaConf.register_new_resolver("cmaxgt0", lambda s: C_max(s) > 0)
OmegaConf.register_new_resolver("not", lambda s: not s)
OmegaConf.register_new_resolver(
    "cmaxgt0orcmaxgt0", lambda a, b: C_max(a) > 0 or C_max(b) > 0
)
# ======================================================= #


def C_max(value: Any) -> float:
    if isinstance(value, int) or isinstance(value, float):
        pass
    else:
        value = config_to_primitive(value)
        if not isinstance(value, list):
            raise TypeError("Scalar specification only supports list, got", type(value))
        if len(value) == 3:
            value = [0] + value
        assert len(value) == 4
        start_step, start_value, end_value, end_step = value
        value = max(start_value, end_value)
    return value


@dataclass
class ExperimentConfig:
    name: str = "default"
    description: str = ""
    seed: int = 0
    device: str = "cuda"

    use_tqdm: bool = True
    exp_root_dir: str = "outputs"

    checkpoint: Optional[str] = None

    scene: str = ""
    camera: dict = field(default_factory=dict)

    prompt_processor: dict = field(default_factory=dict)
    guidance: dict = field(default_factory=dict)

    trainer: dict = field(default_factory=dict)


@dataclass
class RuntimeContext:
    seed: int
    device: torch.device
    dr_generator: dr.random.Generator
    torch_generator: torch.Generator


def create_runtime(seed: int, device: str) -> RuntimeContext:
    torch_device = torch.device(device)
    mi.set_variant("cuda_ad_rgb" if "cuda" in device else "llvm_ad_rgb")


    random.seed(seed)
    torch.manual_seed(seed)
    if torch_device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch_generator = torch.Generator(device=torch_device.type).manual_seed(seed)
    dr_generator = dr.rng(seed)
    return RuntimeContext(
        seed=seed, device=torch_device, 
        torch_generator=torch_generator, 
        dr_generator=dr_generator
    )


class Configurable:
    @dataclass
    class Config:
        pass

    def __init__(self, cfg: Optional[dict] = None, runtime: Optional[RuntimeContext] = None) -> None:
        super().__init__()
        self.cfg = parse_structured(self.Config, cfg)
        self.runtime = runtime


def load_config(*yamls: str, cli_args: list = [], from_string=False, **kwargs) -> Any:
    if from_string:
        yaml_confs = [OmegaConf.create(s) for s in yamls]
    else:
        yaml_confs = [OmegaConf.load(f) for f in yamls]
    cli_conf = OmegaConf.from_cli(cli_args)
    cfg = OmegaConf.merge(*yaml_confs, cli_conf, kwargs)
    OmegaConf.resolve(cfg)
    assert isinstance(cfg, DictConfig)
    scfg = parse_structured(ExperimentConfig, cfg)
    return scfg


def config_to_primitive(config, resolve: bool = True) -> Any:
    return OmegaConf.to_container(config, resolve=resolve)


def dump_config(path: str, config) -> None:
    with open(path, "w") as fp:
        OmegaConf.save(config=config, f=fp)


def parse_structured(fields: Any, cfg: Optional[Union[dict, DictConfig]] = None) -> Any:
    scfg = OmegaConf.structured(fields(**cfg))
    return scfg
