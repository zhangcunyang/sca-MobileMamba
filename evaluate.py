import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

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


def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total

def evaluate_per_class(model, dataloader, device, num_classes=256):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # 打印分类报告
    print("\n=== Per-Class Classification Report ===")
    print(classification_report(all_labels, all_preds, labels=np.arange(num_classes), zero_division=0))

    # 可选：返回混淆矩阵（可视化用）
    cm = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))
    return cm


def compute_log_prob(output, plaintext_byte, key_guess):
    """
    给定模型输出、明文 byte、假设的密钥猜测，返回 log 概率。
    """
    # sbox output = Sbox(p ⊕ k)
    sbox_table = np.array([
        99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
        202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
        183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
        4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
        9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
        83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
        208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
        81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
        205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
        96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
        224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
        231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
        186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
        112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
        225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
        140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22
    ])
    label = sbox_table[plaintext_byte ^ key_guess]
    return output[label].item()  # 返回这个 label 的概率（softmax 前）

def evaluate_guessing_entropy(model, dataloader, plaintexts, real_key, device, target_byte=2, max_traces=1000):
    model.eval()
    key_guesses = np.zeros(256)
    ge_curve = []

    trace_count = 0

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            outputs = model(x)
            outputs = torch.nn.functional.log_softmax(outputs, dim=1).cpu().numpy()

            for i in range(x.size(0)):
                if trace_count >= max_traces:
                    return ge_curve  # 提前返回
                pt = plaintexts[trace_count]

                for k in range(256):
                    label = SBOX[pt ^ k]
                    key_guesses[k] += outputs[i][label]

                rank = np.argsort(-key_guesses).tolist().index(real_key) + 1
                ge_curve.append(rank)

                trace_count += 1  # 手动记录使用了多少 trace

    return ge_curve


def check_label_consistency(h5_path, label_array, target_byte=2, sample_num=100 ):
    """
    验证 label_array 是否等于 Sbox(plaintext ⊕ key)
    h5_path: ASCAD .h5 文件路径
    label_array: 已加载的标签（例如 train_Y）
    target_byte: 哪一个明文/密钥字节位
    sample_num: 随机检查前多少个样本
    """
    import h5py
    import numpy as np

    sbox_table = np.array([
        99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,
        202,130,201,125,250,89,71,240,173,212,162,175,156,164,114,192,
        183,253,147,38,54,63,247,204,52,165,229,241,113,216,49,21,
        4,199,35,195,24,150,5,154,7,18,128,226,235,39,178,117,
        9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,
        83,209,0,237,32,252,177,91,106,203,190,57,74,76,88,207,
        208,239,170,251,67,77,51,133,69,249,2,127,80,60,159,168,
        81,163,64,143,146,157,56,245,188,182,218,33,16,255,243,210,
        205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,
        96,129,79,220,34,42,144,136,70,238,184,20,222,94,11,219,
        224,50,58,10,73,6,36,92,194,211,172,98,145,149,228,121,
        231,200,55,109,141,213,78,169,108,86,244,234,101,122,174,8,
        186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,
        112,62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,
        225,248,152,17,105,217,142,148,155,30,135,233,206,85,40,223,
        140,161,137,13,191,230,66,104,65,153,45,15,176,84,187,22
    ])

    mismatch_count = 0

    start_idx = 50000

    with h5py.File(h5_path, "r") as f:

            for i in range(min(sample_num, len(label_array))):
                    md_idx = start_idx + i
            plaintext = f['metadata'][md_idx][0][target_byte]
            key = f['metadata'][md_idx][1][target_byte]
            expected_label = sbox_table[plaintext ^ key]

            if label_array[i] != expected_label:
                print(f"[Mismatch] idx={i}: label={label_array[i]}, expected={expected_label}")
                mismatch_count += 1

    if mismatch_count == 0:
        print(f"✅ 标签与 Sbox(p ⊕ k) 完全一致（检查了前 {sample_num} 个样本）")
    else:
        print(f"❌ 标签不一致：有 {mismatch_count}/{sample_num} 个样本不匹配")

    return mismatch_count
