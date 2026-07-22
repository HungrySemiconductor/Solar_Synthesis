# plot_loss_from_log.py
import re
import matplotlib.pyplot as plt
import numpy as np

def parse_training_log(log_file_path):
    """从训练日志中提取所有batch loss"""
    with open(log_file_path, 'r', encoding='utf-8') as f:  # 指定utf-8编码
        content = f.read()
    
    # 匹配 loss 值，格式: loss=数字
    pattern = r'loss=(\d+\.\d+)'
    losses = [float(x) for x in re.findall(pattern, content)]
    
    # 匹配 epoch 平均值
    avg_pattern = r'Epoch\s+(\d+)\s+complete\s+\|\s+Avg loss:\s+([\d\.]+)'
    avg_losses = [(int(e), float(l)) for e, l in re.findall(avg_pattern, content)]
    
    return losses, avg_losses

def plot_losses(losses, avg_losses):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. 所有batch loss（原始）
    ax1 = axes[0, 0]
    ax1.plot(losses, 'b-', alpha=0.5, linewidth=0.5, label='Batch Loss')
    ax1.set_xlabel('Batch Index')
    ax1.set_ylabel('Loss')
    ax1.set_title('All Batches (20 epochs × 230 batches)')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    ax1.legend()
    
    # 2. 去掉异常值后的loss（loss < 10）
    ax2 = axes[0, 1]
    filtered = [l for l in losses if l < 10]
    ax2.plot(filtered, 'g-', alpha=0.5, linewidth=0.5)
    ax2.set_xlabel('Batch Index (filtered loss<10)')
    ax2.set_ylabel('Loss')
    ax2.set_title(f'Filtered (removed {len(losses)-len(filtered)} outliers)')
    ax2.grid(True, alpha=0.3)
    
    # 3. 每个epoch的平均loss
    ax3 = axes[1, 0]
    epochs = [e for e, _ in avg_losses]
    avg_vals = [l for _, l in avg_losses]
    ax3.plot(epochs, avg_vals, 'r-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Avg Loss')
    ax3.set_title('Average Loss per Epoch')
    ax3.grid(True, alpha=0.3)
    
    # 4. 滑动平均（窗口=50）
    ax4 = axes[1, 1]
    window = 50
    smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
    ax4.plot(smoothed, 'purple', linewidth=1.5)
    ax4.set_xlabel('Batch Index')
    ax4.set_ylabel('Loss (Smoothed)')
    ax4.set_title(f'Smoothed Loss (window={window})')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_loss_full.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 统计信息
    print(f"\n{'='*50}")
    print(f"统计摘要")
    print(f"{'='*50}")
    print(f"总batch数: {len(losses)}")
    outliers = [l for l in losses if l > 10]
    print(f"异常值(>10): {len(outliers)} ({len(outliers)/len(losses)*100:.1f}%)")
    print(f"最大loss: {max(losses):.2f}")
    print(f"最小loss: {min(losses):.4f}")
    print(f"最终loss (最后10个batch平均): {np.mean(losses[-10:]):.4f}")

if __name__ == "__main__":
    # 使用方法 - 改成你的日志文件名
    log_file = "training_log.txt"  # 把日志内容保存到这个文件
    losses, avg_losses = parse_training_log(log_file)
    plot_losses(losses, avg_losses)