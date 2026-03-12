import os
import random


root_dir = './data/kvasir-seg'
image_dir = os.path.join(root_dir, 'images')


images = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
images.sort()

# 去掉后缀，只留文件名 (例如 "cju0qkwl35piu0993l0de1le.jpg" -> "cju0qkwl35piu0993l0de1le")
ids = [os.path.splitext(f)[0] for f in images]

random.seed(42)
random.shuffle(ids)

# --- 划分比例 8:1:1 ---
total = len(ids)
train_end = int(total * 0.8)
val_end = int(total * 0.9)

train_ids = ids[:train_end]
val_ids = ids[train_end:val_end]
test_ids = ids[val_end:]

print(f"总图片数: {total}")
print(f"训练集: {len(train_ids)} | 验证集: {len(val_ids)} | 测试集: {len(test_ids)}")

def save_txt(id_list, filename):
    save_path = os.path.join(root_dir, filename)
    with open(save_path, 'w') as f:
        for i in id_list:
            f.write(i + '\n')
    print(f"已保存: {save_path}")

save_txt(train_ids, 'train.txt')
save_txt(val_ids, 'val.txt')
save_txt(test_ids, 'test.txt')