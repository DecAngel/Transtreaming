# Transtreaming Codebase
## Install
- Install dependency
```shell
conda env create -f environment.yaml
```
- Download Argoverse-HD and put inside ./data

## Run
- training
```shell
python src/train.py experiment=xxx
```
- testing
```shell
python src/test.py experiment=xxx
```
- sap testing
```shell
python src/sap.py experiment=xxx
```

Detailed configurations are in ./configs
