# Transtreaming Codebase
Code for [**Transtreaming: Adaptive Delay-aware Transformer for Real-time Streaming Perception**](https://doi.org/10.1609/aaai.v39i10.33105)

Note: We are currently refactoring our work to build our streaming perception pipeline on ROS for a better simulation 
of real-world autonomous driving scenarios(end-to-end delay adaptation).

## Install
- Install dependency
```shell
conda env create -f environment.yaml
```
- Download Argoverse-HD inside ./data (from https://mtli.github.io/streaming/)

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

## Checkpoints (with config and logs)
- [Transtreaming S](https://drive.google.com/file/d/1OfvAcdvV60RfQOslg2EZZN3p6xlDje0T/view?usp=sharing)
- [Transtreaming_M](https://drive.google.com/file/d/10Jgge_1P2HKIkrgysEFUkxEDI3eIuSuz/view?usp=sharing)
- [Transtreaming_L](https://drive.google.com/file/d/1mq19Rw6FdHhCghyO4M9TXW5w8n4FvnzC/view?usp=sharing)
