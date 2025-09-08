import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict
import pandas as pd

# 导入原有的SBOX定义
SBOX = [
    99,124,119,123,242,107,111,197, 48,  1,103, 43,254,215,171,118,
    202,130,201,125,250, 89, 71,240,173,212,162,175,156,164,114,192,
    183,253,147, 38, 54, 63,247,204, 52,165,229,241,113,216, 49, 21,
      4,199, 35,195, 24,150,  5,154,  7, 18,128,226,235, 39,178,117,
      9,131, 44, 26, 27,110, 90,160, 82, 59,214,179, 41,227, 47,132,
     83,209,  0,237, 32,252,177, 91,106,203,190, 57, 74, 76, 88,207,
    208,239,170,251, 67, 77, 51,133, 69,249,  2,127, 80, 60,159,168,
     81,163, 64,143,146,157, 56,245,188,182,218, 33, 16,255,243,210,
    205, 12, 19,236, 95,151, 68, 23,196,167,126, 61,100, 93, 25,115,
     96,129, 79,220, 34, 42,144,136, 70,238,184, 20,222, 94, 11,219,
    224, 50, 58, 10, 73,  6, 36, 92,194,211,172, 98,145,149,228,121,
    231,200, 55,109,141,213, 78,169,108, 86,244,234,101,122,174,  8,
    186,120, 37, 46, 28,166,180,198,232,221,116, 31, 75,189,139,138,
    112, 62,181,102, 72,  3,246, 14, 97, 53, 87,185,134,193, 29,158,
    225,248,152, 17,105,217,142,148,155, 30,135,233,206, 85, 40,223,
    140,161,137, 13,191,230, 66,104, 65,153, 45, 15,176, 84,187, 22
]

class EnhancedEvaluator:
    """增强的评估器，支持猜测熵分析和t-SNE可视化"""
    
    def __init__(self, model, device, num_classes=256):
        self.model = model
        self.device = device
        self.num_classes = num_classes
        self.sbox_table = np.array(SBOX)
        
    def compute_per_class_guessing_entropy(self, dataloader, plaintexts, real_key, 
                                         target_byte=2, max_traces=1000):
        """
        计算每个分类的猜测熵
        返回每个类别的平均猜测熵
        """
        self.model.eval()
        
        # 为每个类别收集数据
        class_data = defaultdict(list)  # {class_id: [guessing_entropy_values]}
        
        trace_count = 0
        key_guesses = np.zeros(256)
        
        with torch.no_grad():
            for x, _ in dataloader:
                if trace_count >= max_traces:
                    break
                    
                x = x.to(self.device)
                outputs = self.model(x)
                outputs = torch.nn.functional.log_softmax(outputs, dim=1).cpu().numpy()
                
                for i in range(x.size(0)):
                    if trace_count >= max_traces:
                        break
                        
                    pt = plaintexts[trace_count]
                    
                    # 计算当前trace对应的真实类别
                    true_class = self.sbox_table[pt ^ real_key]
                    
                    # 累加密钥猜测分数
                    for k in range(256):
                        label = self.sbox_table[pt ^ k]
                        key_guesses[k] += outputs[i][label]
                    
                    # 计算真实密钥的排名
                    rank = np.argsort(-key_guesses).tolist().index(real_key) + 1
                    
                    # 将猜测熵添加到对应类别
                    class_data[true_class].append(rank)
                    
                    trace_count += 1
        
        # 计算每个类别的平均猜测熵
        class_ge = {}
        for class_id, ge_values in class_data.items():
            if ge_values:  # 确保有数据
                class_ge[class_id] = np.mean(ge_values)
        
        return class_ge
    
    def identify_poor_classes(self, class_ge, threshold=50):
        """
        识别分不出来的类别（猜测熵较高的类别）
        threshold: 猜测熵阈值，超过此值的类别被认为是"分不出来"的
        """
        poor_classes = []
        for class_id, ge in class_ge.items():
            if ge > threshold:
                poor_classes.append((class_id, ge))
        
        # 按猜测熵降序排列
        poor_classes.sort(key=lambda x: x[1], reverse=True)
        return poor_classes
    
    def extract_layer_features(self, dataloader, layer_name='blocks', max_samples=1000):
        """
        提取指定层的特征用于t-SNE可视化
        """
        self.model.eval()
        features = []
        labels = []
        
        # 注册hook来提取中间层特征
        def hook_fn(module, input, output):
            # 将特征展平并存储
            feat = output.detach().cpu().numpy()
            if len(feat.shape) > 2:  # 如果是多维的，进行全局平均池化
                feat = np.mean(feat, axis=tuple(range(2, len(feat.shape))))
            features.append(feat)
        
        # 找到目标层并注册hook
        target_layer = None
        for name, module in self.model.named_modules():
            if layer_name in name:
                target_layer = module
                break
        
        if target_layer is None:
            raise ValueError(f"找不到层: {layer_name}")
        
        hook = target_layer.register_forward_hook(hook_fn)
        
        try:
            sample_count = 0
            with torch.no_grad():
                for x, y in dataloader:
                    if sample_count >= max_samples:
                        break
                    x = x.to(self.device)
                    _ = self.model(x)
                    labels.extend(y.cpu().numpy())
                    sample_count += x.size(0)
            
            # 合并所有特征
            if features:
                all_features = np.concatenate(features, axis=0)
                return all_features[:max_samples], np.array(labels[:max_samples])
            else:
                return None, None
                
        finally:
            hook.remove()
    
    def plot_tsne_for_classes(self, features, labels, target_classes, 
                            save_path="tsne_visualization.png", 
                            perplexity=30, n_iter=1000):
        """
        对指定类别进行t-SNE可视化
        """
        if features is None or len(features) == 0:
            print("没有特征数据可供可视化")
            return
        
        # 筛选目标类别的数据
        mask = np.isin(labels, target_classes)
        filtered_features = features[mask]
        filtered_labels = labels[mask]
        
        if len(filtered_features) == 0:
            print(f"没有找到类别 {target_classes} 的数据")
            return
        
        print(f"开始t-SNE降维，数据点数量: {len(filtered_features)}")
        
        # 执行t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, 
                   random_state=42, verbose=1)
        tsne_results = tsne.fit_transform(filtered_features)
        
        # 创建可视化
        plt.figure(figsize=(12, 8))
        
        # 为每个类别使用不同颜色
        unique_classes = np.unique(filtered_labels)
        colors = plt.cm.tab20(np.linspace(0, 1, len(unique_classes)))
        
        for i, class_id in enumerate(unique_classes):
            mask = filtered_labels == class_id
            plt.scatter(tsne_results[mask, 0], tsne_results[mask, 1], 
                       c=[colors[i]], label=f'Class {class_id}', 
                       alpha=0.7, s=50)
        
        plt.title(f't-SNE Visualization for Classes: {target_classes}')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"t-SNE可视化已保存到: {save_path}")
    
    def plot_class_guessing_entropy(self, class_ge, save_path="class_guessing_entropy.png"):
        """
        绘制每个类别的猜测熵柱状图
        """
        if not class_ge:
            print("没有猜测熵数据可供绘制")
            return
        
        classes = list(class_ge.keys())
        ge_values = list(class_ge.values())
        
        plt.figure(figsize=(15, 6))
        bars = plt.bar(classes, ge_values, alpha=0.7)
        
        # 为高猜测熵的类别使用不同颜色
        threshold = np.percentile(ge_values, 75)  # 使用75分位数作为阈值
        for i, (bar, ge) in enumerate(zip(bars, ge_values)):
            if ge > threshold:
                bar.set_color('red')
                bar.set_alpha(0.8)
        
        plt.title('每个类别的平均猜测熵')
        plt.xlabel('类别ID')
        plt.ylabel('平均猜测熵')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"类别猜测熵图已保存到: {save_path}")
    
    def comprehensive_analysis(self, dataloader, plaintexts, real_key, 
                             target_byte=2, max_traces=1000, 
                             ge_threshold=50, save_dir="analysis_results"):
        """
        综合分析：计算猜测熵、识别问题类别、进行t-SNE可视化
        """
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        print("开始综合分析...")
        
        # 1. 计算每个类别的猜测熵
        print("1. 计算每个类别的猜测熵...")
        class_ge = self.compute_per_class_guessing_entropy(
            dataloader, plaintexts, real_key, target_byte, max_traces)
        
        # 2. 识别分不出来的类别
        print("2. 识别分不出来的类别...")
        poor_classes = self.identify_poor_classes(class_ge, ge_threshold)
        
        print(f"发现 {len(poor_classes)} 个分不出来的类别:")
        for class_id, ge in poor_classes[:10]:  # 显示前10个最差的
            print(f"  类别 {class_id}: 平均猜测熵 = {ge:.2f}")
        
        # 3. 绘制猜测熵图
        print("3. 绘制类别猜测熵图...")
        self.plot_class_guessing_entropy(
            class_ge, f"{save_dir}/class_guessing_entropy.png")
        
        # 4. 对最差的几个类别进行t-SNE可视化
        if poor_classes:
            print("4. 对分不出来的类别进行t-SNE可视化...")
            worst_classes = [class_id for class_id, _ in poor_classes[:5]]  # 取前5个最差的
            
            # 提取特征
            features, labels = self.extract_layer_features(
                dataloader, layer_name='blocks', max_samples=2000)
            
            if features is not None:
                self.plot_tsne_for_classes(
                    features, labels, worst_classes,
                    f"{save_dir}/tsne_worst_classes.png")
            else:
                print("无法提取特征进行t-SNE可视化")
        
        # 5. 保存详细结果
        results_df = pd.DataFrame([
            {'class_id': class_id, 'guessing_entropy': ge}
            for class_id, ge in class_ge.items()
        ])
        results_df = results_df.sort_values('guessing_entropy', ascending=False)
        results_df.to_csv(f"{save_dir}/class_analysis_results.csv", index=False)
        
        print(f"分析完成！结果保存在 {save_dir}/ 目录中")
        return class_ge, poor_classes
