import sys
import os
import torch
import torchvision.transforms as transforms
import yaml
from contextlib import contextmanager
import pickle

# 将项目根目录添加到 sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(PROJECT_ROOT)

try:
    from datasets import EvalDataset
except ImportError:
    # 尝试在 sys.path 中添加项目根目录后再次导入
    sys.path.append(PROJECT_ROOT)
    from datasets import EvalDataset

@contextmanager
def change_dir(destination):
    try:
        cwd = os.getcwd()
        os.chdir(destination)
        yield
    finally:
        os.chdir(cwd)

def _ensure_traj_names_file(data_folder: str, data_split_folder: str, traj_names_filename: str = "traj_names.txt"):
    """
    BaseDataset 依赖 data_split_folder 下的 traj_names.txt。
    一些数据集的 split 目录可能只有 predefined_index（pkl），没有 traj_names 文件；
    这里在缺失时自动生成，避免加载失败。
    """
    os.makedirs(data_split_folder, exist_ok=True)
    traj_names_path = os.path.join(data_split_folder, traj_names_filename)
    if os.path.exists(traj_names_path):
        return traj_names_path

    # Try to infer trajectory dirs from data_folder
    trajs = []
    for name in sorted(os.listdir(data_folder)):
        p = os.path.join(data_folder, name)
        if not os.path.isdir(p):
            continue
        # Only keep folders that look like trajectories
        if os.path.exists(os.path.join(p, "traj_data.pkl")):
            trajs.append(name)

    if not trajs:
        raise FileNotFoundError(f"No trajectory folders found under {data_folder} (missing traj_data.pkl?)")

    with open(traj_names_path, "w", encoding="utf-8") as f:
        f.write("\n".join(trajs) + "\n")
    return traj_names_path

def get_recon_dataset(
    data_root="/home/payneli/data/nav_datasets",
    split_dir="/home/payneli/project/nwm/data_splits/recon",
    split="test",
    context_size=4,
    image_size=(128, 128)
):
    """
    加载 Recon 数据集 (EvalDataset)
    """
    
    # 硬编码参数以匹配预生成的 pkl 文件
    min_dist_cat = -64
    max_dist_cat = 64
    len_traj_pred = 64
    traj_stride = 1
    
    # 图像变换
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # 数据集分割路径
    data_split_folder = os.path.join(split_dir, split)
    _ensure_traj_names_file(os.path.join(data_root, "recon"), data_split_folder, "traj_names.txt")
    
    # 需要切换到项目根目录，因为 BaseDataset 读取 config/data_config.yaml 是相对路径
    with change_dir(PROJECT_ROOT):
        dataset = EvalDataset(
            data_folder=os.path.join(data_root, "recon"),
            data_split_folder=data_split_folder,
            dataset_name="recon",
            image_size=image_size,
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            len_traj_pred=len_traj_pred,
            traj_stride=traj_stride,
            context_size=context_size,
            transform=transform,
            traj_names="traj_names.txt",
            normalize=True
        )
    
    return dataset

def get_go_stanford_dataset(
    data_root="/home/payneli/data/nav_datasets",
    split_dir="/home/payneli/project/nwm/data_splits/go_stanford",
    split="test",
    context_size=4,
    image_size=(128, 128),
    eval_type: str = "rollout",  # "rollout" | "time" | "full"
):
    """
    加载 go_stanford 数据集（用于语义级 OOD）。

    - eval_type="rollout"/"time": 使用 predefined index（pkl），样本量固定（例如 150/500）。
    - eval_type="full": 不使用 predefined index，走与 recon 相同的逻辑：
        traj_names.txt + traj_data.pkl -> BaseDataset 自动构建 index_to_data（样本量通常更大）。
    """
    min_dist_cat = -64
    max_dist_cat = 64
    len_traj_pred = 64
    traj_stride = 1

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    data_split_folder = os.path.join(split_dir, split)
    data_folder = os.path.join(data_root, "go_stanford")

    # Ensure required traj_names file exists
    _ensure_traj_names_file(data_folder, data_split_folder, "traj_names.txt")

    predefined_index = None
    if eval_type and eval_type != "full":
        predefined_index = os.path.join(data_split_folder, f"{eval_type}.pkl")
        if not os.path.exists(predefined_index):
            raise FileNotFoundError(f"predefined_index not found: {predefined_index}")

    with change_dir(PROJECT_ROOT):
        dataset = EvalDataset(
            data_folder=data_folder,
            data_split_folder=data_split_folder,
            dataset_name="go_stanford",
            image_size=image_size,
            min_dist_cat=min_dist_cat,
            max_dist_cat=max_dist_cat,
            len_traj_pred=len_traj_pred,
            traj_stride=traj_stride,
            context_size=context_size,
            transform=transform,
            traj_names="traj_names.txt",
            normalize=True,
            predefined_index=predefined_index,
            goals_per_obs=1,
        )
    return dataset

if __name__ == "__main__":
    # 测试代码
    try:
        ds = get_recon_dataset()
        print(f"Dataset length: {len(ds)}")
        if len(ds) > 0:
            item = ds[0]
            print("Item shapes (idx, obs_image, pred_image, actions, delta):")
            for x in item:
                print(x.shape)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()



