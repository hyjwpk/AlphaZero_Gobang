import re
import matplotlib.pyplot as plt

# 读取日志文件内容
with open("train.log", "r", encoding="utf-8") as f:
    log_text = f.read()

# 提取 loss 值
losses = [float(m.group(1)) for m in re.finditer(r"loss:([\d.]+)", log_text)]

# 绘图
plt.plot(range(0, len(losses)), losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png")