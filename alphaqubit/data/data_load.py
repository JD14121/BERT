# 第一行必须是os环境变量修复，在所有导入之前
import os
# Windows 补HOME环境变量，解决KeyError
if os.name == "nt":
    os.environ["HOME"] = os.path.expanduser("~")

# 此时环境已装好包，才能正常导入
from fundrive import Zenodo, BaiduPan

def zenodo_direct_to_baidu(doi: str, pan_folder: str):
    pan = BaiduPan()
    pan.login()
    record = Zenodo(doi)
    print(f"数据集标题：{record.title}")
    print(f"文件数量：{len(record.files)}")
    # local_save_path=None 完全不落地本地，直存网盘
    record.download_all(
        drive=pan,
        save_path=pan_folder,
        skip_exist=True,
        local_save_path=None
    )
    print("云端转存完成，文件仅保存在百度网盘")

if __name__ == "__main__":
    target_doi = "10.5281/zenodo.13273331"
    save_dir = "/Zenodo/QEC_13273331"
    zenodo_direct_to_baidu(target_doi, save_dir)