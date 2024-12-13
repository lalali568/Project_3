import matplotlib.pyplot as plt
from docutils.nodes import label
from holoviews.plotting.bokeh.styles import alpha
from matplotlib.backends.backend_pdf import  PdfPages
from datetime import datetime
import os

from sympy.printing.pretty.pretty_symbology import line_width


def plot_loss(config,train_loss):
    """
    绘制训练过程中的损失值
    :param config: dict
    :param train_loss: List
    :return:
    """
    date=datetime.now()
    date=date.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir=f"plot/{config['model']}/{config['dataset']}/loss_record/"
    #创建save_dir文件夹
    os.makedirs(save_dir,exist_ok=True)
    save_path=save_dir+f'{date}_'+'loss.pdf'
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(10, 6))
        plt.plot(train_loss)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title(f'{config['model']}-{config['dataset']}-Training Loss')

        plt.grid(True, linestyle='--',alpha=0.5)
        pdf.savefig()
        plt.close()

def plot_pre_result(config, recon_data, ori_data, recon_loss, labels):
    """
    绘制重建结果,以及重构损失
    :param config: 配置字典，包含模型和数据集信息
    :param recon_data: 重构数据 (array-like)
    :param ori_data: 原始数据 (array-like)
    :param recon_loss: 重构损失 (array-like)
    """
    # 获取当前时间戳用于文件命名
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # 创建保存目录
    save_dir = os.path.join("plot", config['model'], config['dataset'], "recon_plot")
    os.makedirs(save_dir, exist_ok=True)

    # 保存路径
    save_path = os.path.join(save_dir, f"{date}_recon.pdf")

    # 使用 PdfPages 保存图表
    with PdfPages(save_path) as pdf:
        for i in range(recon_data.shape[-1]):
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

            # 绘制原始数据和重构数据

            ax1.plot(recon_data[:, i], label="Reconstructed Data", color="red", linewidth=0.5)
            ax1.plot(ori_data[:, i], label="Original Data", color="black", linewidth=0.5)

            y_min = min(recon_data[:, i].min(), ori_data[:, i].min())
            y_max = max(recon_data[:, i].max(), ori_data[:, i].max())

            ax1.fill_between(range(len(recon_data)), y_min, y_max, where=(labels == 1), alpha=0.3, color="green")
            ax1.legend(loc="upper right")
            ax1.set_title(f'{config["model"]} - {config["dataset"]} - Dimension {i}')

            # 绘制重构损失
            ax2.plot(recon_loss[:, i], label="Reconstruction Loss", color="blue", linewidth=0.5)
            ax2.fill_between(range(len(recon_loss)), 0, recon_loss[:, i].max(),where=(labels == 1), alpha=0.3, color="green")
            ax2.legend(loc="upper right")
            ax2.set_title(f'Reconstruction Loss - Dimension {i}')

            # 调整布局
            plt.tight_layout()
            pdf.savefig(fig)  # 保存当前图表到 PDF
            plt.close(fig)  # 关闭当前图表，释放内存

def plot_total_loss(config,total_loss,labels):
    """
    绘制每个时间点的损失
    :param config: 配置字典，包含模型和数据集信息
    :param
    """
    date=datetime.now()
    date=date.strftime("%Y-%m-%d_%H-%M-%S")
    save_dir=f"plot/{config['model']}/{config['dataset']}/total_loss/"
    #创建save_dir文件夹
    os.makedirs(save_dir,exist_ok=True)
    save_path=save_dir+f'{date}_'+'total_loss.pdf'
    with PdfPages(save_path) as pdf:
        plt.figure(figsize=(10, 6))
        plt.plot(total_loss)
        plt.fill_between(range(len(total_loss)), 0, total_loss.max(),where=(labels == 1), alpha=0.3, color="green")
        plt.xlabel('time')
        plt.ylabel('Loss')
        plt.title(f'{config['model']}-{config['test_data_path']}-Test Loss')

        plt.grid(True, linestyle='--',alpha=0.5)
        pdf.savefig()
        plt.close()
