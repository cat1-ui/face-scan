"""1:N 人脸匹配器。

核心职责（实施方案 M4 T4-1 ~ T4-5）：
- gallery 缓存为实例属性（只加载一次，不每帧查库）
- matmul 一次算出 query 与所有样本的余弦相似度
- top1 / top2 提取
- 拒识分支（低于阈值报「未识别」）
- 相似脸 gap 判断（top1 - top2 < 0.08 → 人工确认）

关键点：
- 所有特征已 L2 归一化，gallery @ query 即余弦相似度。
- 只加载 face_status='active' 的人。
"""
import numpy as np

from core import config
from repository.feature_repo import FeatureRepo
from repository.person_repo import PersonRepo


class Matcher:
    """人脸匹配器，持有 gallery 缓存。"""

    def __init__(self):
        self.feature_repo = FeatureRepo()
        self.person_repo = PersonRepo()
        self.gallery = None          # (N, 128) float32
        self.person_ids = None       # (N,) 每个样本对应的 person_id
        self.load_gallery()

    def load_gallery(self):
        """从数据库加载所有 active 人员的特征到内存。

        每次录入成功后调用刷新。
        """
        items = self.feature_repo.list_all_active()
        if not items:
            self.gallery = np.zeros((0, config.FEATURE_DIM), dtype=np.float32)
            self.person_ids = np.zeros((0,), dtype=np.int64)
            return
        ids, feats = zip(*items)
        self.person_ids = np.asarray(ids, dtype=np.int64)
        self.gallery = np.stack([np.asarray(f, dtype=np.float32)
                                 for f in feats])

    def match(self, query_feature):
        """对 query 特征做 1:N 匹配。

        返回 dict：
            top1_person_id / top1_sim / top2_sim / recognized / need_confirm
        """
        q = np.asarray(query_feature, dtype=np.float32)
        if self.gallery.shape[0] == 0:
            return {
                "top1_person_id": None,
                "top1_sim": 0.0,
                "top2_sim": 0.0,
                "recognized": False,
                "need_confirm": False,
            }

        # 一次 matmul 算所有相似度
        sims = self.gallery @ q          # (N,)
        order = np.argsort(sims)[::-1]   # 降序
        top1_idx = order[0]
        top1_sim = float(sims[top1_idx])
        top1_person = int(self.person_ids[top1_idx])

        # top2：取「top1 所属 person 之外」的最高相似度，
        # 避免同一人的多个样本互相比较导致 gap 误判。
        top2_sim = 0.0
        for idx in order:
            if int(self.person_ids[idx]) != top1_person:
                top2_sim = float(sims[idx])
                break

        # 拒识分支
        if top1_sim < config.SIMILARITY_THRESHOLD:
            return {
                "top1_person_id": None,
                "top1_sim": top1_sim,
                "top2_sim": top2_sim,
                "recognized": False,
                "need_confirm": False,
            }

        # gap 判断（相似脸：不同人之间的相似度接近）
        need_confirm = (top1_sim - top2_sim) < config.GAP_THRESHOLD

        return {
            "top1_person_id": top1_person,
            "top1_sim": top1_sim,
            "top2_sim": top2_sim,
            "recognized": True,
            "need_confirm": need_confirm,
        }

    def person_name(self, person_id):
        p = self.person_repo.get(person_id)
        return p["name"] if p else "未知"
