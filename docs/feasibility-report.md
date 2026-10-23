# NspireLM 可行性研究报告：在 TI-Nspire CX II CAS 上离线运行 Transformer 语言模型

调研日期：2026-10-23。估算脚本：[`tools/estimate.py`](../tools/estimate.py)，输出存档：[`docs/estimate-output.txt`](estimate-output.txt)。

**标注约定**

- **【实测】**：有人在真机上测过，并给出了来源。
- **【文档】**：来自官方文档、源码或社区资料。
- **【推算】**：本报告自己的计算，假设写在旁边。
- **【未核实】**：没有找到来源。

---

## 0. 一页摘要

**结论：可行，而且已经有人做过一部分。** 有两点和最初的假设不同：

1. **原生 Transformer 的先例已经存在。** ChatTLM（[alex-messerlian/nspire-tlm](https://github.com/alex-messerlian/nspire-tlm)，附预印本 *Letting the Tools Do the Math*）在 TI-Nspire CX II CAS 上用 Ndless + C 跑通了：
   - 模型：10.9M 参数，llama2.c 风格，int8（group 88）。
   - 速度：电池供电、396 MHz 下，decode **1.88 tok/s**；53 个 token 的 prefill 用时 24.3 s。【实测】
   - 所以"Nspire 上没有本地 Transformer"这个前提不成立。但还没有人做过 **15M 以上**的模型，ChatTLM 的 16.6M 配置就因为内存不够没能装载。【实测】
2. **任务里说的"84K 参数 Nspire 模型"没有找到。** TI-Planet、Cemetech、GitHub、Reddit 和 Hackaday 都检索过。最接近的是 CalcGPT 3：它用 TI-Python 跑一个 int8 的词级 GRU，但没有标注参数量，也没有真机速度数据（[Cemetech](https://www.cemetech.net/downloads/files/3024/x4069)）。这一条标为【未核实】。

**瓶颈在哪里。** 按 ChatTLM 的实测数据推算：

- 以"可交互"（≥0.5 tok/s）为目标时，**先撞到的是内存容量**，不是算力。Ndless 程序单块 malloc 最多约 21.5 MiB，分块总计约 27 MiB。【实测】
- 以 ≥2 tok/s 为目标时，**算力**变成瓶颈。

**推荐配置**

| 阶段 | 模型 | 量化 | 内存占用 | 预期速度 |
|---|---|---|---|---|
| MVP | 稠密 Llama 风格，约 12–18M 参数（d=384, L=8, vocab 4096, ctx 512） | int8 权重 + int8 KV cache | 约 15–20 MiB | **1.5–3 tok/s**【推算】 |
| 进阶 | 稠密约 30–40M 参数 | int4（group 32/64） | 约 20–25 MiB | **0.8–2.5 tok/s**【推算】 |
| 展示版 | 稠密核心 15–20M（RAM）+ 25–70M 的 PLE 查找表（RAM 或闪存） | 核心 int8/int4，PLE int4 | RAM 约 25 MiB + 闪存 ≤ 65 MiB | 1–2.5 tok/s；PLE 放闪存时每 token 再加约 58 ms【推算】 |

三值（BitNet b1.58）放进 RAM 的理论容量最大（约 98M），但 1–50M 规模下质量损失明显（[Spectra](https://arxiv.org/abs/2407.12327)：三值模型要到 1B 以上才追平同规模模型），**不建议作为主线**。

**最值得做的应用：** "模型理解自然语言 → 确定性引擎做精确计算"。ChatTLM 已经证明这条路行得通：给出正确关系式时，物理题正确率 96.7%（【实测】，[FACTS.md](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/paper/FACTS.md)）。能否直接调用 TI 自带的 CAS 引擎是**最大的未知项**：目前没有找到公开的 Ndless→CAS API。

---

## 1. 硬件与系统参数表

| # | 项目 | 值 | 来源 | 置信度 |
|---|---|---|---|---|
| H1 | SoC / CPU 核 | NS2018 SoC，**ARM926EJ-S**（CP15 Main ID `0x41069265`） | [Zephray 超频文章](https://www.zephray.me/post/on_the_way_to_overclock_nspire_cxii/)；ChatTLM [HARDWARE.md](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/HARDWARE.md)；[Wikipedia](https://en.wikipedia.org/wiki/TI-Nspire_series) | 实测 |
| H2 | 指令集 | **ARMv5TEJ**（含 DSP 扩展 "E" 和 Jazelle） | [ARM DDI0198](https://support.arm.com/documentation/ddi0198/e/introduction/about-the-arm926ej-s-processor) | 文档 |
| H3 | 主频 | 电池供电 **396 MHz**（×33）。插 USB 后降到 **288 MHz**，要重启才恢复 | [NoverII 页面](https://tiplanet.org/forum/archives_voir.php?id=2671594)；Firebird `core/cx2.cpp`；ChatTLM 电池测试确认 396 MHz | 实测（有冲突，见注 1） |
| H4 | 总线 | AHB 198 MHz，APB 99 MHz | [Hackspire CX II I/O](https://hackspire.org/index.php?title=Memory-mapped_I/O_ports_on_CX_II)；Firebird | 文档 |
| H5 | 旧款对比 | CX 为 132 MHz，初代 Nspire 为 90/120 MHz | [Wikipedia](https://en.wikipedia.org/wiki/TI-Nspire_series)；[Nover 3](https://www.ticalc.org/archives/files/fileinfo/434/43428.html) | 文档 |
| H6 | FPU / VFP / SIMD | **没有**。Ndless 工具链用 `--with-float=soft` 编译，`nspire-gcc` 只传 `-mcpu=arm926ej-s`。ChatTLM 也说明 "no floating-point unit"。没有 ARMv6 的 SIMD（SMLAD、SXTB16 等） | [Ndless build_toolchain.sh](https://github.com/ndless-nspire/Ndless/blob/master/ndless-sdk/toolchain/build_toolchain.sh)；[LLVM ARMParallelDSP](https://llvm.org/docs/doxygen/ARMParallelDSP_8cpp.html) | 强间接证据（TI 没有官方声明） |
| H7 | 可用 DSP 指令 | SMULxy / SMLAxy（16×16+32）、SMULWy / SMLAWy（32×16）、SMLALxy、QADD / QSUB / QDADD / QDSUB、CLZ、PLD、LDRD / STRD | [ARM9EJ-S TRM DDI0222B 表 8-2](https://documentation-service.arm.com/static/5e8e476f88295d1e18d3aecb) | 文档 |
| H8 | 指令周期（零等待存储，TRM） | 见下方周期表 | 同上；[GCC arm926ejs.md](https://gnu.googlesource.com/gcc.git/+/refs/heads/master/gcc/config/arm/arm926ejs.md) | 文档 |
| H9 | L1 cache | **I 16 KB + D 8 KB**，4 路，32 B 行（CP15 Cache Type `0x1D112152`，另用指针追逐延迟扫描交叉验证：8 KB 到 16 KB 之间延迟从 15 ns 跳到 211 ns）。没有 L2，TCM 为 0 | ChatTLM HARDWARE.md A10–A12（CX II 真机）；[Hackspire Hardware](https://hackspire.org/index.php/Hardware)（旧款同为 16+8） | 实测 |
| H10 | 片上 SRAM | 256 KB @ `0xA4000000`；另有 LCD 扫描用 SRAM @ `0xA8000000`。Ndless 程序能不能用前者：**未核实** | Hackspire CX II I/O；本组织 NspireAI 仓库 `docs/test-status.md`（LCD 只从 `0xA8000000` 扫描） | 文档 / 部分实测 |
| H11 | RAM | 64 MB LPDDR @ `0x10000000` | [TI 官方规格](https://education.ti.com/en/products/calculators/graphing-calculators/ti-nspire-cx-ii-cx-ii-cas/specifications)；Hackspire | 文档 |
| H12 | **Ndless 单块 malloc 上限** | **22,576,128 B ≈ 21.5 MiB**（刚复位后测得）。另有一次测得 22,609,920 B。程序崩溃后掉到 **5.02 MiB**，复位前不会恢复 | ChatTLM [FACTS.md §6](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/paper/FACTS.md)、HARDWARE.md B1 | 实测 |
| H13 | 分块 malloc 总量 | 64 KB 块共 **28,442,624 B ≈ 27.1 MiB**。两块实测至少 22.10 MiB 可装（26.87 MiB 装不下）。4 次装载/释放循环后没有碎片 | 同上 B2、§6 | 实测 |
| H14 | DRAM 顺序读带宽 | **70.5 MB/s**，memcpy **50.9 MB/s**（见注 1） | ChatTLM HARDWARE.md B4/B5 | 实测（时钟有争议） |
| H15 | 存储 | 128 MB SPI NAND，TI 标称用户可用 "90+ MB"。某台机器上 `nsp info` 显示存储 96,862,208 B，其中 62,560,256 B 空闲 | TI 规格；Zephray；ChatTLM HARDWARE.md | 文档 / 实测 |
| H16 | 闪存文件读取 | 顺序 **1.46 MB/s**；随机 4 KB 读取 **58 ms** | ChatTLM HARDWARE.md B8/B9 | 实测 |
| H17 | 单文件大小限制 | 没有找到明确限制。Zehn 可执行文件最大 16 MB，所以权重应放在单独的数据文件里 | [Hackspire Zehn](https://www.hackspire.org/Zehn) | **未核实**（数据文件） |
| H18 | 超频 | NoverII（critor，2020-12，alpha）：作者自己那台到了 **492 MHz（×41）**，CoreMark 1050。设置不持久，插 USB 或关机就复位。没有官方安全范围 | [NoverII](https://tiplanet.org/forum/archives_voir.php?id=2671594)；[Zephray](https://www.zephray.me/post/on_the_way_to_overclock_nspire_cxii/) | 文档（安全范围未核实） |
| H19 | Ndless 支持的 CX II OS | r2015：5.2.0.771；r2017：5.3.0.564；r2020：6.2.0.333；**r2022：6.4.0.74**（2026 年发布） | [Ndless releases](https://github.com/ndless-nspire/Ndless/releases)；[ndless.me](https://ndless.me/)；[Cemetech](https://www.cemetech.net/forum/viewtopic.php?t=20563) | 文档 |
| H20 | 最新 OS | **6.4.0.74**。6.3 不能降级，也不受 Ndless 支持，只能升到 6.4.0.74 再装 Ndless。不能降到低于已装版本 | [TI 下载页](https://education.ti.com/en/software/details/en/2726D6FC251C40648FDEE5AFABED3B95/TI-NspireCXII_OS)；r2022 release notes | 文档 |
| H21 | 考试模式（Press-to-Test） | TI 官方：考试期间已有程序和文档被锁定，退出后恢复。"Ndless 程序在 PTT 下不能运行"只有一条 2012 年的论坛说法 | [TI 知识库 22151](https://education.ti.com/en/customer-support/knowledge-base/ti-nspire-family/product-usage/22151)；[Omnimaga](https://omnimaga.org/ti-nspire-projects/pttkiller-enable-access-to-documents-in-ptt-mode) | 部分核实 |
| H22 | TI-Python | 基于 MicroPython（报告为 3.4.0），**heap 约 2.07 MB，最大连续块 683 KB**。2020 年 TI-Planet 测试中，它是所有计算器里最快的 | [TI-Planet t=24224](https://tiplanet.org/forum/viewtopic.php?t=24224) | 实测（社区） |
| H23 | Lua | Lua 5.1（OS 3.0 起），**没有 `io`/`os` 库**，权重只能嵌在文档里 | Wikipedia；Hackspire Lua 页 | 部分核实（OS 3.x 时期的资料） |
| H24 | 屏幕 / SDK | 320×240、16 位色。新版硬件是旋转 90° 的 240×320 面板（HW-W），genzehn 需要 `--240x320-support true`。SDK 自带 nspire-io、zlib、freetype2、nSDL（SDL 1.2）；nGL 是一个独立的定点 3D 库 | TI 规格；[genzehn.cpp](https://github.com/ndless-nspire/Ndless/blob/master/ndless-sdk/tools/genzehn/genzehn.cpp)；[Ndless SDK](https://github.com/ndless-nspire/Ndless)；[nGL](https://github.com/Vogtinator/nGL) | 文档 |
| H25 | 计时 | SDK 没有计时 syscall，要直接读 SP804 定时器 MMIO（`0x90010000`，APB 时钟） | Hackspire CX II I/O；ChatTLM HARDWARE.md §C | 文档 |

**ARM9EJ-S 关键指令周期**（[TRM 表 8-2](https://documentation-service.arm.com/static/5e8e476f88295d1e18d3aecb)，假设 cache 命中）

| 指令 | 周期 | 说明 |
|---|---|---|
| 数据处理指令 | 1 | 寄存器控制移位时为 2 |
| LDR | 1 | 下一条就用结果时 +1；字节或半字结果 +2 |
| LDM（n 个寄存器） | n | 每周期约 1 个字，是读权重最快的方式 |
| MUL / MLA | 2 | 结果被下一条使用时为 3；累加链不停顿 |
| **SMLAxy / SMULxy / SMLAWy** | **1** | 结果被下一条使用时为 2；累加链例外 |
| SMLALxy | 2 | |
| SMULL / SMLAL | 3 | |
| B / BL | 3 | |
| PLD | 1 | ARM926 的 cache 会不会真的预取：**未核实** |

单发射、5 级流水线，理论峰值是每周期 1 次 16×16 MAC，即约 396 M MAC/s。

**注 1：ChatTLM 测量数据之间的冲突。** HARDWARE.md 正文写着当时 CPU "只有 198/144 MHz"，并且按 144 MHz 修正了 B4、B10 等数值。同一个文件的页眉和后来的 FACTS.md 又写着电池复测确认是 396 MHz。所以 70.5 MB/s 这个带宽值，可能低估了 1.375 倍（未修正的原始值是 97 MB/s）。naive C 的 int8 MAC 速度（34.9 MMAC/s，4.08 周期/MAC）同样受影响。**这两个数必须在你自己的机器上重测**（见 §6）。下文的端到端估算只用 FACTS.md 里的 396 MHz 整机数据，那部分是自洽的。

**注 2：本组织 NspireAI 仓库里的"约 60 KB 装载预算"。** 那是常驻 / NGC 启动路径下 Zehn 镜像分配的经验上限（见 NspireAI `docs/test-status.md` 2026-09-29 条目），不是普通 `main()` 程序 malloc 的上限。ChatTLM 的 21.5 MiB 是普通程序实测得到的。两者不冲突，但 NspireLM 应该按普通程序启动，不要走常驻路径。

### 1.1 不越狱的两条路线（对比用）

| 路线 | 能做到什么 | 上限估计 |
|---|---|---|
| TI-Python（MicroPython） | heap 2 MB，最大连续块 683 KB；权重放在 `bytes` 里，最多约 0.5–0.6M 个 int8 参数。解释器每次 MAC 大约要几 µs 到几十 µs【推算】 | 约 50K 参数模型每 token 0.1–1 s【推算】。CalcGPT 3（GRU，TI-Python）是现成先例，但没有真机速度数据 |
| Lua | 没有文件 IO，权重要以字符串嵌进 .tns 文档；heap 大小没找到数据 | 能跑几十 K 参数的玩具模型，速度和 Python 同级或更慢【推算/未核实】 |
| **Ndless C/asm（主线）** | 约 27 MiB 堆，可直接写 MMIO、手写汇编 | 见 §2 |

---

## 2. 理论上限计算

### 2.1 校准数据（全部为 396 MHz 电池实测，ChatTLM [FACTS.md §4–7](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/paper/FACTS.md)）

**整机代价模型**

- 每 token 时间 ≈ **32.3 ms + 37.5 ms × N**（N 为百万参数数，6 层，位置 11–18 附近，R² 0.9998）。
- 换算：每秒约 26.7 M "参数·MAC"，约 14.8 周期/参数。

**各阶段时间占比**

| 阶段 | 占比 |
|---|---|
| FFN | 56.7% |
| QKV | 16.5% |
| 分类器 | 10.6% |
| attention | 9.0% |
| RoPE | 5.0% |

matmul 合计约 84%。

**随位置衰减。** 填满的 cache 下，pos 0 时 2.42 tok/s，pos 511 时只剩 0.80 tok/s，下降 67%。原因是 ChatTLM 的 **KV cache 是 fp32**，attention 里用软浮点。这一块有很大的优化空间。

**宽度扫描**

| dim | 参数 | 内存（文件 + KV） | tok/s @pos 11–18 | tok/s @pos 259–266 |
|---|---|---|---|---|
| 176 | 3.1M | 7.26 MiB | 6.82 | 2.72 |
| 264 | 6.4M | 12.70 MiB | 3.64 | 1.82 |
| 352 | 10.9M | 19.34 MiB | 2.27 | 1.18 |
| 440 | 16.6M | 27.19 MiB | 装不下 | — |

**和奔腾 II 交叉校准**（[EXO Labs](https://blog.exolabs.net/day-4/)，【实测】）：Win98 P-II 跑 fp32 的 stories15M 是 1.03 tok/s。这个模型每 token 约 15M MAC，其中 9.2M 是 32000 词表分类器，要读 60 MB 权重，也就是每秒约 15.5M MAC。Nspire 上 ChatTLM 的 C 引擎是每秒约 26.7M 参数·MAC，int8 权重只需读 1/4 的字节。所以 **"Nspire 跑 15M 模型和奔腾 II 速度相当，甚至略快"**，与下面的估算一致：stories15M 在 Nspire int8 上约 1.7 tok/s【推算】。

另一个参照是 ARMv5TE 同族的 Nintendo DS（ARM946E-S，67 MHz，软浮点）：stories260K 跑 1.78 tok/s（[nintendo-ds-llm](https://github.com/RileyGreiff/nintendo-ds-llm)）。只按频率换算，在 396 MHz 上约 10 tok/s。改用整数后，按代价模型估计约 24 tok/s，可见**必须避免软浮点**。

### 2.2 内存容量上限【推算】

假设：预留 3 MiB 给激活、int8 KV、tokenizer、UI 和栈。group 量化的 scale 开销为 int8 每 64 个权重一个 fp32，int4 每 64 个一个 fp16，三值每 256 个一个 fp16。

| 量化 | 单块 21.5 MiB | 分块 27.1 MiB |
|---|---|---|
| int8 | **18M** | **24M** |
| int4 | **37M** | **48M** |
| 三值（2 bit 打包） | 75M | 98M |

### 2.3 每 token 计算量与速度【推算】

稠密模型每 token 的 MAC 数约为 N（非 embedding 参数加分类器），attention 再加 2·L·ctx·d。三档估算：

- **C-int8**：直接套用 ChatTLM 的代价模型（保守）。
- **2×-int8**：手写汇编 matmul，加上 int8 KV 去掉软浮点 attention。假设单参数代价减半，**这是推荐的规划值**。
- **asm 上限**：取计算（int8 / int4 / 三值分别按 2.5 / 3.0 / 2.0 周期/MAC，假设值）和 DRAM 流式读取（70.5 MB/s）中的较大者，再加 32 ms 固定开销。这是理想值。

| 参数量 | C-int8 | 2×-int8 | asm-int8 | asm-int4 | asm-三值 |
|---|---|---|---|---|---|
| 0.26M | 23.8 | 26.9 | 27.6 | 29.2 | 29.7 |
| 3M | 6.9 | 11.3 | 12.9 | 18.2 | 21.1 |
| 7M | 3.4 | 6.1 | 7.3 | 11.7 | 14.8 |
| 11M | 2.25（ChatTLM 实测 2.27） | 4.2 | 5.1 | 8.7 | 11.4 |
| 15M | 1.68 | 3.2 | 3.9 | 6.9 | 9.3 |
| 20M | 1.28 | 2.5 | 3.0 | 5.4 | 7.5 |
| 25M | 1.03 | 2.0 | 2.4 | 4.5 | 6.3 |
| 45M | 0.58 | 1.14 | 1.4 | 2.7 | 3.9 |
| 90M | 0.29 | 0.58 | 0.7 | 1.4 | 2.1 |

单位为 tok/s，对应短上下文。上下文长时速度会下降：ChatTLM 在 pos 256 降了约 48%；改成 int8 KV 后预计降幅明显变小。

**"可交互"（≥0.5 tok/s）前提下的实际上限**

- 单看算力：C-int8 可以到约 53M，2×-int8 约 105M，asm-int4 约 260M。
- 但**内存只能装下 int8 约 24M、int4 约 48M**。所以 0.5 tok/s 时，**约束是内存，不是算力**。
- 实际上限：**稠密 int8 约 20M，int4 约 40–45M**，速度约 1–2.5 tok/s。
- 如果目标是 2 tok/s：C-int8 约 12M，2×-int8 约 25M。

**int4 是否值得做？** ChatTLM 的 HARDWARE.md 认为"低于 int8 对吞吐没有帮助"。它的依据是 144 MHz 下 naive C 的 4.08 周期/MAC，那时 CPU 处理权重比 DRAM 送数据还慢。如果手写汇编能做到 2.5 周期/MAC 左右，int8 就会变成**受带宽限制**（396M/2.5 = 158 M MAC/s，DRAM 只能送约 70 M 字节/s），这时 int4 能直接提速。**结论取决于两个尚未可靠测量的数：真实 DRAM 带宽，以及 asm 内核的周期/MAC**（见 §6 基准清单）。

### 2.4 权重从闪存流式读取【推算，基于实测 1.46 MB/s】

- 要保持 0.5 tok/s，每 token 只能从闪存读约 2.9 MB：int8 约 2.75M 参数，int4 约 5.5M。**稠密流式没有意义。**
- 把整个约 90 MB 用户空间都塞满权重（int4 约 169M 参数，三值约 349M），每 token 要 **约 62 s**，即 0.016 tok/s。这和 EXO Labs 在 P-II 上从硬盘读 1B 模型的 0.0093 tok/s 是同一个量级，只能当作"能跑"的噱头展示。

### 2.5 PLE（per-layer embedding）查找表【推算】

esp32-ai（[slvDev/esp32-ai](https://github.com/slvDev/esp32-ai)，【文档】）的配置：

- 总参数 28.9M，其中稠密核心只有 556K（6 层，d=96）。
- PLE 表 32768 × 6 × 128 ≈ 25.2M，int4 存放在内存映射的闪存里。
- ESP32-S3 上 9.88 tok/s（[CNX](https://www.cnx-software.com/2026/08/03/28-9m-parameter-llm-runs-locally-on-esp32-s3-at-9-tokens-s)）。

在 Nspire 上，要把**同一个 token 在所有层的 PLE 向量排成一行连续存放**。这样每 token 只做**一次**随机读取，读出 400 B–1 KB。

ChatTLM 判定 "PLE is dead"，是因为它假设每层各读一次（6 次 × 58 ms = 350 ms）。改成连续一行后只需约 58 ms。这一点要在真机上验证；如果 NAND 驱动的 4 KB 读取有缓存，还会更快。

| 配置 | 行大小 | 词表 | 位置 | 表大小 | PLE 参数 | 每 token 额外时间 |
|---|---|---|---|---|---|---|
| L6, d_ple 128 | 396 B | 32768 | RAM | 12.4 MiB | 25.2M | ≈0 |
| L8, d_ple 128 | 528 B | 32768 | RAM | 16.5 MiB | 33.6M | ≈0 |
| L8, d_ple 128 | 528 B | 65536 | NAND | 33.0 MiB | 67.1M | +58 ms |
| L8, d_ple 256 | 1040 B | 65536 | NAND | 65.0 MiB | 134.2M | +58 ms |

**总参数量能推多高**

- **全放 RAM**：int4 稠密约 20M（10 MiB）加 PLE 约 25–34M（12–16 MiB），合计 **约 45–55M**，速度约 2–4 tok/s。
- **PLE 放闪存**：稠密 int8 约 18M 在 RAM，PLE 67–134M 在 NAND，合计 **约 85–150M**，速度约 1.5–2.5 tok/s。

**注意：** PLE 参数不参与计算，"参数量"作为宣传口径好看，但效果远不如同等规模的稠密模型。它主要增强的是按 token 存的"词汇知识"。

PLE 大词表的配套问题：输入词表大（32K–64K）只影响 PLE 表和输入 embedding。**输出分类器的维度要另外控制**（esp32-ai 的输出词表是 25,353），否则分类器的计算量会吃掉速度。

### 2.6 KV cache、上下文和词表【推算】

**KV cache = 2 × L × ctx × d × 每元素字节数**

| 配置 | ctx 256 | ctx 512 | ctx 1024 |
|---|---|---|---|
| L6 d352（ChatTLM，fp32） | 4.1 MiB | **8.2 MiB**（实测占总量 43%） | 16.5 MiB |
| L8 d384，int8 | 1.5 MiB | **3.0 MiB** | 6.0 MiB |
| L8 d512，int8 | 2.0 MiB | 4.0 MiB | 8.0 MiB |
| L12 d512，int8 | 3.0 MiB | 6.0 MiB | 12.0 MiB |

**KV 必须改成 int8**（每头每位置一个 scale）。只这一项，在 d352/ctx512 配置下就能腾出约 6 MiB，大约够多放 6M 个 int8 参数。

**词表**：embedding 和分类器都是 V × d。

| 词表 | d | 参数 | 说明 |
|---|---|---|---|
| 128（字符级） | 384 | 0.05M | |
| 4096 | 384 | 1.57M | |
| 32000 | 288 | 9.2M | stories15M 的 61%，分类器每 token 9.2M MAC |

**推荐 2K–4K 的 BPE 词表，并且与输出层权重共享（tied）。**

---

## 3. 先例对照表

| 项目 | 设备（CPU / RAM） | 模型与参数 | 量化 | 速度 | 真离线 | 链接 |
|---|---|---|---|---|---|---|
| **ChatTLM** | **TI-Nspire CX II CAS**，ARM926EJ-S 396 MHz，OS 6.4.0.74 | Llama 风格（源自 llama2.c `runq`），10.9M，d352 L6 H8，tok4096 | int8 g88 | **1.88 tok/s** decode | 是 | [GitHub](https://github.com/alex-messerlian/nspire-tlm) |
| "84K 参数 Nspire 模型" | — | — | — | — | — | **未找到 / 未核实** |
| CalcGPT 3 | Nspire CX II，TI-Python | 词级 GRU，vocab 640，hidden 30 | int8 打包 | 没有真机数据 | 是 | [Cemetech](https://www.cemetech.net/downloads/files/3024/x4069)；[GitHub](https://github.com/Muntinator/Muntchat/releases/tag/v3.1) |
| TI-GPT-Nspire | Nspire CX，Ndless 客户端 | 云端 LLM，经 USB 代理 | — | — | **否** | [GitHub](https://github.com/SuperManzDev/TI-GPT-Nspire) |
| ChatLink | Nspire CX | 云端（Claude API） | — | — | 否 | [GitHub](https://github.com/DroppingAnvil/ChatLink) |
| NspireAI（本组织） | CX II CAS | 云端 LLM，USB/NavNet 转发 | — | — | 否 | 同组织仓库 |
| TI-32 | TI-84 Plus + ESP32-C3 | ChatGPT 中转 | — | — | 否 | [GitHub](https://github.com/ChromaLock/TI-32) |
| calcGPT | TI-84 Plus CE + 电脑串口 | OpenAI API | — | — | 否 | [GitHub](https://github.com/novano1d/calcGPT) |
| MNIST CNN | TI-84 Plus CE，eZ80 48 MHz | 4 层 CNN | ? | 约 18 s/次推理 | 是 | [Cemetech](https://www.cemetech.net/forum/viewtopic.php?p=309757) |
| HERMES OPTIMUS | TI-84 Plus SE，TI-BASIC | 4-60-12 MLP 拼写纠错 | — | — | 是 | [TechEBlog](https://www.techeblog.com/?p=215557) |
| Casio fx-CG50 / NumWorks / HP Prime 上的本地语言模型 | — | 没有找到 | — | — | — | 未核实 |
| **nintendo-ds-llm** | **NDS Lite，ARM946E-S（ARMv5TE）67 MHz**，4 MB | stories260K；1.2M 字符级模型 | Q8_0 权重，软浮点计算 | 260K：1.78 tok/s；1.2M：约 1.4 tok/s | 是 | [GitHub](https://github.com/RileyGreiff/nintendo-ds-llm) |
| llama98.c | Pentium II 约 350 MHz，128 MB，Win98 | 260K / 15M / Llama 3.2 1B | fp32 | 39.31 / 1.03 / 0.0093 tok/s（1B 是用分片和磁盘基准外推的） | 是 | [EXO 博客](https://blog.exolabs.net/day-4/)；[GitHub](https://github.com/exo-explore/llama98.c) |
| dosllam2 | 486DX2-66 / P-MMX 200 / P-III 667 | 260K–110M | fp32 | 486：2.08（260K）；MMX-200：15.3 / 0.43（15M）；P-III：80 / 2.35 / 0.89（42M） | 是 | [博客](https://yeokhengmeng.com/2025/04/llama2-llm-on-dos/) |
| llama2.c64 | C64 + 2 MB REU | stories260K | fp32 | 约 8 分钟/token | 是 | [GitHub](https://github.com/ytmytm/llama2.c64) |
| llama2-psp | PSP，MIPS 333 MHz，64 MB | stories260K | fp32 | 35–36 tok/s | 是 | [GitHub](https://github.com/caiomadeira/llama2-psp) |
| psvita-llm | PS Vita，Cortex-A9 超频到 555 MHz | 260K / 15M | （可能是 fp32） | 约 120 / 约 1.8 tok/s | 是 | [GitHub](https://github.com/callbacked/psvita-llm) |
| iiama2.c | Wii，PowerPC 729 MHz | stories15M | int8 | CPU 12.7 tok/s，GPU 61.9 tok/s | 是 | [GitHub](https://github.com/brysontang/iiama2.c) |
| esp32-llm | ESP32-S3 240 MHz + PSRAM | stories260K | fp32 | 19.13 tok/s | 是 | [GitHub](https://github.com/DaveBben/esp32-llm) |
| ESP32-Tiny-LLM | ESP32-S3 / ESP32 | stories260K | fp32 | 22 / 2.32 tok/s（2.32 为无 PSRAM、从闪存读） | 是 | [GitHub](https://github.com/Circuit-Digest/ESP32-Tiny-LLM) |
| **esp32-ai** | ESP32-S3，512 KB SRAM + 8 MB PSRAM + 16 MB 闪存 | TinyStories，总 28.9M（稠密 556K + PLE 25.2M） | int4 g128，int8 激活 | 9.88 tok/s | 是 | [GitHub](https://github.com/slvDev/esp32-ai)；[CNX](https://www.cnx-software.com/2026/08/03/28-9m-parameter-llm-runs-locally-on-esp32-s3-at-9-tokens-s) |
| PicoLlama | RP2350 + 8 MB PSRAM | 14 MB 模型 | ? | 约 850 ms/token | 是 | [GitHub](https://github.com/earlephilhower/PicoLlama) |
| MCUBERT | 商用 MCU，<256 KB | BERT-tiny/mini | int8 | — | 是 | [arXiv 2410.17957](https://arxiv.org/abs/2410.17957) |
| bitnet.cpp | 桌面 CPU | BitNet b1.58，最大到 100B | 三值 | 100B 约 5–7 tok/s | 是 | [arXiv 2410.16144](https://arxiv.org/abs/2410.16144) |

GBA（ARM7TDMI）、诺基亚功能机、iPod、Palm、STM32 上的 llama2.c 移植：**没有找到**，GBA 在 [awesome-llm-on-anything](https://github.com/emrehangorgec/awesome-llm-on-anything) 的 "Wanted" 列表里。

**对本项目最有用的结论**

1. ChatTLM 给出了同一台机器上的直接标定曲线，规划时直接沿用它的代价模型。
2. 软浮点是 ARMv5 上最大的陷阱：NDS 在 67 MHz 下只有 1.78 tok/s；ChatTLM 的长上下文衰减也来自 fp32 KV。
3. 在 fp32 的 x86/MIPS 移植里，15M 比 260K 慢 30–40 倍。用 int8 加小词表可以明显缩小这个差距。

---

## 4. 推荐技术方案

### 4.1 架构取舍（针对 ARM926EJ-S）

| 架构 | 优点 | 缺点 | 建议 |
|---|---|---|---|
| 标准 Llama 风格 Transformer（RMSNorm + RoPE + SwiGLU，可加 GQA） | 训练生态成熟（llama2.c 可直接用），ChatTLM 已经验证可行 | KV cache 随 ctx 线性增长，attention 代价也随位置增长 | **主线**，配合 int8 KV、GQA 和 ctx 512 |
| RWKV（v6/v7） | 每 token 计算量恒定，没有 KV cache，状态只有 O(L·d) | 有 exp、sigmoid 等非线性，定点化麻烦；小规模公开权重少；没有找到 MCU 移植先例 | 第二阶段的实验项 |
| 线性注意力 / Mamba | 长上下文开销恒定 | 扫描是逐元素运算，乘加比例低；数值敏感，int8 难做 | 不推荐 |
| 无注意力 MLP / n-gram 混合 | 极简单、极快 | 质量差 | 只适合当 baseline |
| 字符级小模型 | 词表极小，分类器几乎不占时间；不需要分词器 | 序列长 2–4 倍，等于把速度打了折扣 | 适合窄领域（表达式、化学式），要和小 BPE 比较 |

### 4.2 量化：在没有 SIMD、没有 FPU 的 ARMv5 上怎么做最快

**int8 权重 × int16 激活（MVP）**【推算 / 设计】

- 激活每 token 量化成 int16，两个一组打包成 32 位字，用 SMLABB / SMLATT。
- int8 权重用**偏移二进制**存放（w+128，无符号）。这样 `AND r, W, #0x00FF00FF`（掩码常驻寄存器）和 `AND r, M, W, LSR #8` 两条指令，就能把 4 个权重拆进两个 16 位字段。
- 拆出来的字段直接配合 SMLAxy 的上下半字使用。最后统一减去 128·Σa（Σa 每行只算一次）。
- 每 4 个 MAC 的成本：1 次 LDR 权重 + 1 次 LDM 激活（摊薄后）+ 2 次 AND + 4 次 SMLAxy ≈ **8–10 周期**，即 2–2.5 周期/MAC。

**int4**

- 一个字装 8 个 nibble。用 `AND 0x000F000F` 加移位拆成 4 组，每组 2 个 16 位字段，每 8 个 MAC 多出约 4 条拆包指令，约 **2.5–3 周期/MAC**。
- 读的字节数减半。如果实测证明 int8 受带宽限制，int4 就能直接提速。

**三值（BitNet b1.58）：能完全用加减代替乘法吗？**

能。每个权重 2 bit（或者 5 bit 编码 3 个权重），对应 +a / −a / 0 三种操作。但 ARMv5 上逐位判断要用条件执行（ADDNE / SUBNE），每个权重约 2 条指令，并不比 SMLAxy 快。更好的做法有两种：

- **T-MAC / bitnet.cpp TL 系列的查表法**：每 2–4 个激活预先算出所有 ±组合的表（9 或 16 项，放在 L1），每个权重组做一次 LDRH 加一次 ADD（[T-MAC](https://arxiv.org/abs/2407.00088)；[bitnet.cpp](https://arxiv.org/abs/2502.11880)）。估计约 **1.5–2 周期/权重**，要实测。
- **SWAR 打包累加**：`MLA acc, x, (w1<<16 | w0)` 一次完成两个"共享激活"的乘加，需要保留护位并定期清空。
- 三值最大的好处是**容量**，RAM 里能放约 98M 参数。但质量问题（§0）让它更适合作为后期实验。

**Group 量化粒度。** ChatTLM 实测 group 32 比 group 88 慢 27%（scale 应用开销）。建议 **group 64–128，scale 用 Q16 定点数**，不要用 fp32。

### 4.3 非线性运算的定点实现

| 运算 | 方案 | 参考 |
|---|---|---|
| softmax | 减去最大值后算 2^x，整数部分移位，小数部分查 256 项表（Q15）；累加用 int32，最后做一次整数除法或倒数。也可以用 I-BERT 的二次多项式 i-exp | [I-BERT](https://arxiv.org/abs/2101.01321)；[CMSIS-NN softmax_q7](https://keil.com/pack/doc/CMSIS_Dev/NN/html/group__Softmax.html) |
| RMSNorm | 平方和用 int64（SMLAL）累加；1/√x 用 CLZ 归一化，查 256 项种子表，再做 1 次 Newton 迭代（Q16） | I-BERT 整数 sqrt；CLZ 见 H7 |
| RoPE | 对每个 pos × head_dim/2 预计算 Q15 的 sin/cos 表（ctx 512、hd 48 时为 512 × 24 × 2 × 2 B = 48 KB）。旋转用 SMULxy 完成 | 【推算】 |
| SiLU / SwiGLU | 门控输入量化到 int8 或 int16 后查表（int8 时 256 项，int16 时 4096 项加线性插值） | 【推算】 |
| 分类器 / 采样 | int32 logits 直接做 argmax（贪心）；top-k 只需对前 k 个做定点 softmax | 【推算】 |

参照 llama2.c 的 `runq.c`：它是 Q8_0 对称 int8，每组一个 fp32 scale，激活在线量化，但 RMSNorm 等仍然是 fp32（[runq.c](https://github.com/karpathy/llama2.c/blob/master/runq.c)）。**Nspire 上要把剩下的 fp32 全部定点化**，这正是 ChatTLM 还没做完的部分。

### 4.4 热点优化

- **matmul 内核用手写 ARM 汇编**：一次处理 4 行输出（4 个累加寄存器），让激活只读一遍。用 LDM 一次读 4–8 个字的权重。LDR 和使用它的 SMLA 之间要隔开，避免 load-use 停顿。
- **激活常驻 cache**：d=384 时 int16 激活约 768 B，FFN 隐层（1024）约 2 KB，都能放进 8 KB 的 D-cache。权重是一次性流过的，没法复用 cache，所以**分块的目的是让激活留在 cache 里，权重顺序流过**。
- 尝试把激活和临时缓冲放进 256 KB 片上 SRAM（`0xA4000000`），**前提是确认 OS 没有占用它**（未核实）。
- **权重打包格式**：按"4 行交错、每行 group 对齐"预先排好，让内核只做顺序读取。scale 单独放一个数组。
- **cache 配置**：ChatTLM 的修复记录提醒过，调用 libndls 的 `clear_cache` 会连带 invalidate。整个推理期间不要改动 cache 和 MMU 设置。
- **时钟**：所有基准都必须在**拔掉 USB、用电池供电**时测。插着 USB 时只有 288 MHz。

### 4.5 工具链与权重导出流程

1. **Ndless SDK**：`git clone --recursive https://github.com/ndless-nspire/Ndless`，运行 `ndless-sdk/toolchain/build_toolchain.sh`（锁定版本：gcc 14.2.0、binutils 2.44、newlib 4.5、gdb 16.2），然后把 `bin` 加入 PATH。
   - 构建链：`nspire-gcc`（实际是 `arm-none-eabi-gcc -mcpu=arm926ej-s`）→ `nspire-ld --emit-relocs` → `genzehn --240x320-support true` → `.tns`。
   - 出处：[Hackspire 入门](https://www.hackspire.org/C_and_assembly_development_introduction/)、[build_toolchain.sh](https://github.com/ndless-nspire/Ndless/blob/master/ndless-sdk/toolchain/build_toolchain.sh)。
   - 本组织 NspireAI 仓库已经有 Docker 化的 Ndless 构建脚本（`scripts/build-program-docker.sh`），可以直接复用。
2. **训练**：用 PyTorch 训练 llama2.c 兼容的模型（`model.py` / `train.py`），做 QAT 或 PTQ。
3. **导出**：写自定义导出脚本，输出 `nslm` 格式。
   - 头部：magic、版本、配置、量化类型。
   - 正文：按 4.4 的交错格式打包好的权重，加上 Q16 scale、RoPE 表和 tokenizer。
   - 所有偏移都按 32 B 对齐。文件名用 `.tns` 结尾（例如 `model.nslm.tns`），否则 TI 的传输软件可能拒收（**未核实**，但这是社区惯例）。
4. **参考实现**：在 PC 上用同一套 C 代码（只是不编进汇编内核）做**逐比特一致**的参考推理，用来和 PyTorch 结果对比 logits 误差。
5. **传输**：用 TI Link 或 [n-link](https://github.com/lights0123/n-link) 把文件拷到 `/documents/ndless/...`。程序启动时调用 `enable_relative_paths(argv)`，然后用 `fopen` 读取。
6. **模拟器调试**：[Firebird](https://github.com/nspire-emus/firebird) 能模拟 CX II（-T/CAS），提供 GDB stub（端口 3333，`arm-none-eabi-gdb -ex "target remote :3333"`）。但它需要从你自己的机器 dump 出 boot1/boot2，CX II 还要额外提取解密密钥。**Firebird 的计时不可信，只用来查正确性**，性能数据必须在真机上测。

### 4.6 输入输出界面

- **显示**：用 nspire-io 或直接写 16 位帧缓冲（320×240 RGB565；HW-W 机型要按 240×320 旋转）。8×8 或 6×8 的点阵字体每屏能放 53×30 或 40×30 个字符。生成过程中**逐 token 流式刷新**，加一个进度条或 tok/s 显示。
- **输入**：用 `isKeyPressed` 扫描键盘，做一行可编辑的输入框，带 Ctrl/Shift 层和常用模板菜单。NspireAI 仓库已经有一套完整的"按 QWERTY 位置布局 + 屏幕图例"方案，可以复用。
- **长推理期间**：每生成 1 个 token 检查一次 Esc 中断。生成结果存为 .tns 文档或剪贴板文本，方便用户贴回 Calculator 页面。
- **中文**：模型本身先做英文或数学符号。中文字体 + IME 的工作量很大，可以参考 NspireAI，但不放进 MVP。

### 4.7 训练端：数据集与规模

| 任务 | 数据 | 建议规模 | 质量预期 |
|---|---|---|---|
| TinyStories 故事生成（技术演示） | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)（212 万篇，7.6 GB） | 3–20M，4K 词表 | 论文显示 <10M 已经能写出流畅的故事，1M 常常重复，2.5M 开始连贯（[arXiv 2305.07759](https://arxiv.org/abs/2305.07759)） |
| 自然语言 → CAS 表达式翻译 | 合成数据：用模板和程序生成"文字描述 → TI 语法表达式"对，用 SymPy 做等价校验 | 5–20M | 窄领域，输出短，适合这个规模（ChatTLM 的同类任务正确率约 90–97%） |
| 物理 / 单位换算的工具调用 | ChatTLM 的数据流程；DeepMind [mathematics_dataset](https://arxiv.org/abs/1904.01557) 的 `measurement__conversion` | 5–15M | 已有实证 |
| 化学方程式配平 | 合成化学式加精确求解器；**配平本身交给整数零空间求解器** | 1–5M（只负责解析化学式） | 模型只做解析，可靠性靠求解器保证 |
| 符号积分 / 解 ODE | [Lample & Charton](https://arxiv.org/abs/1912.01412) 的数据生成方法 | 原论文约 40M 以上（d512 L6）；计算器上只能做小规模的简化版 | 不确定，只作为研究性子课题 |

---

## 5. 应用场景与可行性

| # | 方向 | 内容 | 可行性 | 展示价值 |
|---|---|---|---|---|
| A | **自然语言 → CAS**（模型理解，CAS 计算） | 用户输入 "integrate x^2 sin x from 0 to pi" 或 "解 x^2-5x+6=0"，模型输出 `∫(x^2*sin(x),x,0,π)` 或 `solve(x^2-5x+6=0,x)` | **模型部分高**（输出短，语法封闭，5–15M 足够，可以用约束解码保证语法合法）。**接入部分未知**：没有找到公开的 Ndless→TI CAS API。可选的接法：(1) 逆向 OS 里的 CAS 求值入口（风险高，而且和 OS 版本绑定）；(2) 把输出写成文档或剪贴板文本，由用户粘贴到 Calculator 页面执行（可靠）；(3) 自带一个小型确定性求解器（ChatTLM 的路线） | 最高。"模型负责理解，CAS 负责精确" 的分工本身就是故事 |
| B | **物理 / 化学 / 单位工具调用助手** | 模型从题目里提取已知量和关系式，写出工具调用，由 C 运行时精确计算并配平化学方程式 | **高**，ChatTLM 已经证明 96.7%（给出正确关系式时） | 高，而且是真正有用的离线工具 |
| C | **TinyStories 级生成演示** | 15–20M 稠密或 PLE 大模型在计算器上写小故事 | **高**，纯技术演示 | 中高。"计算器上跑 5000 万参数模型"有话题性，但实用价值低 |
| D | 公式 / 定理检索问答 | 小模型加本地公式库，做意图分类和公式检索 | 中（检索部分可以不用神经网络） | 中 |
| E | TI-Basic / Lua 代码补全 | 帮学生写计算器程序 | 中（需要专门收集语料） | 中 |

**建议排序：B（已有标杆，最快出成果）→ A（输出到文档或剪贴板的版本）→ C（规模展示）。** 直接调用 CAS 作为独立的逆向子课题来做，不要阻塞主线。

**考试合规提醒：** Ndless 本身可能违反考试规则，PTT 模式下程序会被锁定（H21）。项目应该定位为课外演示和学习工具。

---

## 6. MVP 分阶段路线图

| 阶段 | 目标 | 产出 | 完成标准 |
|---|---|---|---|
| P0 基准测试（1–2 周） | 在自己的 CX II CAS 上完成 §7 的清单 | `bench/` 程序、`docs/hardware-measured.md` | 每一项有电池供电、396 MHz 下的测量值，并用定时器交叉校验时钟 |
| P1 移植（1–2 周） | llama2.c `runq` 改为 C 定点版，跑 stories260K | 在 Nspire 上输出和 PC 参考实现 token 一致 | 260K 模型 ≥15 tok/s |
| P2 定点化 | 去掉所有软浮点（RMSNorm、softmax、RoPE、KV） | int8 KV，Q15/Q16 非线性 | 与 PyTorch 的 logits top-1 一致率 ≥99%，长上下文衰减 <30% |
| P3 汇编内核 | int8 matmul 手写汇编，权重交错打包 | `kernels/matmul_q8.S` | 周期/MAC ≤3；15M 级模型 ≥2.5 tok/s |
| P4 自训模型 | 训练 12–18M、4K 词表的任务模型（方向 B 或 A） | `train/`、`export/` | 真机任务正确率达到预设目标 |
| P5 UI | 输入框、流式输出、模板菜单、保存结果 | `.tns` 应用 | 不用电脑，单人可以完成一次问答 |
| P6 扩展（可选） | int4 内核或 PLE 表（RAM，然后 NAND） | 30–50M（稠密 int4）或 50–150M（含 PLE） | ≥1 tok/s |
| P7 研究项（可选） | RWKV 定点版；三值查表内核；Ndless→CAS 调用的逆向 | 实验报告 | — |

---

## 7. 主要风险、未知项与真机基准清单

### 7.1 风险与未知项

| 风险 / 未知项 | 影响 | 处理方式 |
|---|---|---|
| ChatTLM 的时钟和带宽数据自相矛盾（注 1） | 决定 int4 是否值得做 | P0 用 SP804 定时器加 PMU 寄存器交叉校验时钟后重测 |
| 降频：ChatTLM 有一次运行记录到 198/144 MHz | 速度可能掉到 1/2.75 | 长时间运行时监控 PMU 寄存器和定时器，记录温度和电量条件 |
| 崩溃后堆只剩 5 MiB，必须复位 | 演示前必须复位 | 程序启动时先检测可分配量，不够就提示复位 |
| Ndless→CAS 没有公开接口 | 方向 A 的"直连"方案 | 先做输出到文档或剪贴板的版本；逆向单独立项 |
| 片上 SRAM 能否使用 | 优化空间 | 在 P0 实测 |
| PLE 单行随机读 58 ms 是否成立；数据文件有没有大小上限 | P6 | P0 实测大文件（32/64 MB）的 fopen 和 fseek |
| 小规模三值 / int4 的质量损失 | 规模路线 | 在 PC 上先做消融实验 |
| OS 升级导致 Ndless 失效 | 用户群 | 锁定 OS 6.4.0.74，并在文档里说明 |
| Firebird 计时不准 | 调试效率 | 只用它查正确性 |

### 7.2 真机基准测试清单（P0）

所有项目都在**拔掉 USB、电池供电**的条件下测。每项至少 3 次，**刚复位后**测一次，运行一段时间后再测一次。

1. **时钟确认**：读 PMU（`0x90140000`）的倍频，用 SP804 快定时器对照 32.768 kHz 定时器跑 10 s，误差应 <0.5%。
2. **cache**：读 CP15 Cache Type 寄存器，做指针追逐延迟扫描（1 KB–64 MB）。
3. **malloc 上限**：单块二分查找，统计 64 KB 分块总量；连续做 4 次分配/释放循环检查碎片。
4. **片上 SRAM**：`0xA4000000` 区域能否读写；对比 OS 运行前后的内容，判断是否被占用。
5. **DRAM 带宽**：分别测 LDR 循环、LDM 8 字循环、memcpy、写带宽，块大小 64 KB–16 MB。
6. **int8 点积速度**：naive C、偏移二进制 + SMLAxy 的汇编、int4 拆包汇编、三值查表 / 条件加减。报告周期/MAC，包括 cache 内（4 KB）和流式（8 MB）两种情况。
7. **软浮点代价**：float 的乘、加和 `expf` 各要多少周期，用来说明为什么必须去掉。
8. **文件读取**：单个 1/8/32/64 MB 的 `.tns` 数据文件能否传入和打开；顺序 fread 吞吐（块大小 512 B–256 KB）；随机 400 B 和 4 KB 读取的延迟分布。
9. **端到端**：用 stories260K 和 15M 级随机权重模型，测 pos 0 / 128 / 256 / 511 的 tok/s，与本报告 §2.3 的表对比。
10. **持续负载**：连续生成 10 分钟，看 tok/s 是否下降，以及时钟和电量的变化。

---

## 附：主要来源

- ChatTLM / nspire-tlm：<https://github.com/alex-messerlian/nspire-tlm>（[HARDWARE.md](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/HARDWARE.md)、[FACTS.md](https://github.com/alex-messerlian/nspire-tlm/blob/main/docs/paper/FACTS.md)）
- Zephray，*On the way to overclock the TI nspire CX II*：<https://www.zephray.me/post/on_the_way_to_overclock_nspire_cxii/>
- Hackspire CX II MMIO：<https://hackspire.org/index.php?title=Memory-mapped_I/O_ports_on_CX_II>；Hardware：<https://hackspire.org/index.php/Hardware>；Zehn：<https://www.hackspire.org/Zehn>
- Ndless：<https://github.com/ndless-nspire/Ndless>；Firebird：<https://github.com/nspire-emus/firebird>
- ARM9EJ-S TRM（DDI 0222B）：<https://documentation-service.arm.com/static/5e8e476f88295d1e18d3aecb>
- TI-Planet NoverII：<https://tiplanet.org/forum/archives_voir.php?id=2671594>；TI-Python heap：<https://tiplanet.org/forum/viewtopic.php?t=24224>
- llama2.c：<https://github.com/karpathy/llama2.c>；EXO Labs：<https://blog.exolabs.net/day-4/>
- esp32-ai：<https://github.com/slvDev/esp32-ai>
- TinyStories：<https://arxiv.org/abs/2305.07759>；T-MAC：<https://arxiv.org/abs/2407.00088>；bitnet.cpp：<https://arxiv.org/abs/2502.11880>；I-BERT：<https://arxiv.org/abs/2101.01321>；Spectra：<https://arxiv.org/abs/2407.12327>；BitNet b1.58 Reloaded：<https://arxiv.org/abs/2407.09527>

**核实范围：** ChatTLM 的数值是本报告从仓库原始文件中逐项读出的。其余网页内容部分经过抓取工具的摘要，个别原文可能被转述。以上数值引用前都建议点开原链接复核。
