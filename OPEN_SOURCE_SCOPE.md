# Open-source scope

## Included core code
- LlamaGen v3 training, sampling, and evaluation pipeline
- Janus v3 training, sampling, and evaluation pipeline
- Shared evaluation code for COCO / GenEval / HPSv2
- Core runtime modules required by the above scripts
- Sanitized local config template

## Deliberately excluded for now
- Non-v3 experimental branches and older script families
- Analysis notebooks, plotting code, experiment result dumps, logs, wandb artifacts
- Private environment configs, secrets, internal proxies, and absolute internal paths
- Large checkpoints and datasets
