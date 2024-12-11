import os
import shutil

# 原始資料夾和新資料夾的路徑
source_folder = './root_place365_val_train2'  # 原始資料夾的路徑
destination_folder = './rename_root_place365_val_train2'  # 新資料夾的路徑

# 確保新資料夾存在
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# 重新命名和移動圖片
start_index = 0
for i in range(18250, 36500):  # 編號範圍從18250到36499
    old_file = os.path.join(source_folder, f'{i}.png')  # 舊檔案的完整路徑
    if os.path.exists(old_file):
        new_file = os.path.join(destination_folder, f'{start_index}.png')  # 新檔案的路徑
        shutil.copy(old_file, new_file)  # 複製並重命名檔案
        start_index += 1

print("圖片重命名完成！")
