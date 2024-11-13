# Mask-Pack

## Installation
```bash
# only cpu
bash scripts/install.sh -c 0

# use cuda (version: 11.8)
bash scripts/install.sh -c 11.8
```

## Usage example
```bash
python main.py --config_path settings/v1_T-t1-h100-c04-n128-b64-R15.yaml --mode both
```