# AlphaZero-Gobang

本项目基于 [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) 实现，原始版本通过 AlphaZero 方法对 Gomoku（五子棋）进行训练，并在短时间内获得不错的对弈效果。

在此基础上，我进行了以下扩展与优化：

## ✨ 项目增强内容

- ✅ **PyTorch 版 9×9 五子棋模型**：训练了适用于标准五子棋（9×9 棋盘，五子连珠）的模型，兼容原项目推理流程。

- ✅ **并行蒙特卡洛树搜索（Parallel MCTS）**：引入树并行与虚拟损失机制，多进程同时进行模拟搜索，加快推理速度。
- ✅ **性能测试脚本**：提供测试脚本，可一键比较串行与并行 MCTS 的速度差异。

## 🚀 快速开始

✅ mcts_alphaZero 并行性能测试

```bash
test_alphaZero.py
```

**串行 vs 并行 性能对比 （NVIDIA A100 GPU/8进程 加速 4.6x）：**

```
(串行): avg_time = 1.7381s, playout/s = 575.3
(并行): avg_time = 0.3749s, playout/s = 2667.4
```

✅ mcts_pure 并行性能测试

```bash
test_pure.py
```

**串行 vs 并行 性能对比 （8进程 加速 4.5x）：**

```
(串行): avg_time = 2.7728s, playout/s = 360.7
(并行): avg_time = 0.6100s, playout/s = 1639.4
```

## **📂 原始项目说明**

本项目基于以下原始仓库扩展开发：

> 原始项目作者：[@junxiaosong](https://github.com/junxiaosong)

> 仓库地址：https://github.com/junxiaosong/AlphaZero_Gomoku

完整的原始说明请见 [README_original.md](./README_original.md)