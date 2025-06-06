# Execution Trace Replay (et_replay)
`et_replay` is a tool designed for replaying Chakra Execution Traces (ET) from machine learning models.

## Installation
To install `et_replay`, use the following commands:

```bash
$ git clone https://github.com/pytorch-labs/chakra_replay/
$ conda create -n et_replay python=3.10
$ conda activate et_replay
$ cd chakra_replay
$ pip3 install -r requirements.txt
$ pip3 install .
```

## Running et_replay
Unzip tests/inputs/resnet_et.json.gz
```bash
gzip -d tests/inputs/resnet_et.json.gz
```
Replay it with the following command.
```bash
$ python3 -m et_replay.tools.et_replay --input tests/inputs/resnet_et.json -c --profile-replay
```

> Note: When analyzing performance values from et_replay, refer to the collected Kineto traces rather than the execution time reported by et_replay. Kineto traces are only collected when --profile-replay is provided.

## License

Chakra replay is released under Apache-2.0 license. Please see the [`LICENSE`](LICENSE) file for more information.
