# NspireLM

在 TI-Nspire CX II CAS 上完全离线运行 Transformer 语言模型：用 Ndless 原生 C/ARM 汇编实现，不连电脑，不用云端。

- 可行性研究报告：[`docs/feasibility-report.md`](docs/feasibility-report.md)
- 理论上限估算脚本：[`tools/estimate.py`](tools/estimate.py)（`python3 tools/estimate.py`，输出存档于 [`docs/estimate-output.txt`](docs/estimate-output.txt)）

## 当前状态

- AOT 编译器 `compiler/nslmc.py` 加纯整数运行时 `runtime/`（无任何浮点运算），设计见 [`docs/architecture.md`](docs/architecture.md)。
- 质量门：在 stories15M 上与 float32 参考相比，top-1 一致率 98.2%，KL 0.0006，困惑度 2.364 → 2.361。
- Host 构建与 ARMv5TE（arm926ej-s，qemu）构建输出逐比特一致。

```sh
python3 compiler/nslmc.py stories15M.bin tokenizer.bin --out build/stories15M
make MODEL=build/stories15M host arm
build/stories15M/nslm_host build/stories15M/model.nslm build/stories15M/tok.nslt -p "Once upon a time"
MODELS_DIR=~/models VALID=~/data/TinyStories-valid.txt tests/run_all.sh   # 完整验证
```
