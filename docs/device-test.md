# 真机测试指南（TI-Nspire CX II / CX II CAS）

所有能在电脑上验证的都已经验证过（`tests/run_all.sh`）：

- 整数引擎与 float32 参考做质量对比；
- host、ARMv5 汇编、ARMv5 C 三种构建逐比特一致；
- 应用自检在 host 和 qemu ARMv5 上都报 `BITEXACT`；
- 用 ARM926 周期模型预测速度。

还剩下必须在真机上才能确认的几件事：

- 程序能否在 Ndless 下正常启动、显示和响应按键；
- 计算器上算出的 logits 是否与电脑逐比特一致；
- 真实速度、可用内存，以及缓存行填充的形状（周期模型唯一没定下来的参数）。

## 需要的东西

- 能运行 Ndless 的 CX II / CX II CAS（OS 6.2.0.333 或 6.4.0.74，用 Ndless r2022；OS 6.3.x 不支持，需要先升级到 6.4.0.74）；
- TI-Nspire Link / Student Software，或者 [n-link](https://github.com/lights0123/n-link)；
- 本仓库 Release 或 CI 构建产物中的 `nspire/` 目录，里面有三个文件：

| 文件 | 内容 |
|---|---|
| `nslm.tns` | 程序 |
| `model.nslm.tns` | 模型权重（stories15M，词表裁剪到 12750，约 9.7 MB） |
| `tok.nslt.tns` | 分词器 |

## 先装好 Ndless

普通安装方式下 Ndless **不常驻**：关机重启或复位以后都要重新激活一次。

1. 从 <https://github.com/ndless-nspire/Ndless/releases> 下载 r2022，取出 `ndless_installer_4.5.5-6.2.0-6.4.0.tns` 和 `ndless_resources.tns`。
2. 在计算器根目录建一个名叫 `ndless` 的文件夹，把这两个文件放进去。
3. 在计算器上打开安装程序，按任意字母键等待完成。失败或自动重启就再试一次。

## 步骤

1. **把三个文件放进计算器的同一个文件夹**，例如 `My Documents/nslm/`。
2. **拔掉 USB 线。** 插着 USB 时 CPU 只跑 288 MHz，不插是 396 MHz。速度测试要在电池供电下进行。
3. **建议先复位一次，然后重新打开 Ndless 安装程序。** 复位方法：同时按 `doc` + `enter` + `EE`，或者用背面的复位孔。复位能保证内存没有被之前崩溃的程序占住：ChatTLM 实测过，崩溃后堆只剩约 5 MiB，必须复位才能恢复。但复位也会让 Ndless 失效，所以复位后一定要再运行一次 `ndless/ndless_installer...`。
4. **在文件浏览器里打开 `nslm.tns`。** 加载约 9.7 MB 需要 5–8 秒，底部有进度条。
5. **按 `menu` 运行自检。** 屏幕上会列出结果，同时写入同一文件夹下的 `nslm_selftest.txt.tns`。
6. **试着生成：** 输入一个故事开头，比如 `Lily found a`，按 `enter`。
   - `esc`：生成过程中停止；不在生成时退出程序。
   - `tab`：切换温度（0 / 0.8 / 1.0）。
   - `shift`：大写和符号。
7. **把 `nslm_selftest.txt.tns` 传回电脑发给我。** 也可以直接拍照或抄下屏幕上的几行。

## 自检报告怎么看

```
cpu_mhz=396 battery_ok=1
hash=xxxxxxxx expect=xxxxxxxx BITEXACT (... ms)
tok_us_pos8=...  tok_us_last=...
seq_us=... stride_us=... (4 MB)
cycles_per_line seq=... stride=...
free_largest_kb=... state_kb=...
```

| 字段 | 含义 | 预期 |
|---|---|---|
| `cpu_mhz` | 从 PMU 读出的主频 | 396（若是 288，说明 USB 还插着） |
| `BITEXACT` | 计算器上的 logits 哈希与电脑上完全相同 | 必须是 `BITEXACT`；出现 `MISMATCH` 就是 bug |
| `tok_us_pos8` | 位置 8 处生成一个 token 的微秒数 | 周期模型预测 128 ms（7.8 tok/s），可信区间约 128–160 ms |
| `tok_us_last` | 最后一个位置（255）的耗时 | 预测约 151 ms |
| `cycles_per_line seq / stride` | 顺序读与跨行读的每缓存行周期数 | 两者相减可以算出行填充中关键字延迟 `lc` 与每字间隔 `beat` 的分配，这是周期模型唯一未定的参数 |
| `free_largest_kb` | 模型加载后，最大还能 malloc 多少 | 用来决定能否换更大的模型（stories42M） |

## 如果出问题

| 现象 | 可能原因 / 做法 |
|---|---|
| 打开 `nslm.tns` 时提示文档格式不受支持（"This document format is not supported"） | Ndless 没有激活：重新打开 `ndless/ndless_installer...`；如果已经激活过还提示这个，可能是内存不够，复位后重新激活 Ndless 再试 |
| 提示 "not found next to this program" | 三个文件不在同一文件夹，或者扩展名不是 `.tns` |
| 提示 "Not enough memory" | 先复位计算器再试 |
| 屏幕卡住不动 | 拍照记录卡在哪一步（加载进度多少），然后复位 |
| `MISMATCH` | 请把整份报告发给我：这说明计算器上的计算与电脑结果不一致，需要修 |
