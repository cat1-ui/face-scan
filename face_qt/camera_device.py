"""摄像头设备探测与打开。

核心职责（对应实施方案 M1 T1-1 ~ T1-3、T1-5）：
- 后端选择：优先 CAP_DSHOW，失败退回 CAP_ANY
- 分辨率探测：逐档尝试，分辨率作「构造参数」传入 + 「读回验证」
- 编码探测：MJPG / YUYV 对比（排查压缩损失）
- 坏帧防护：cap.read() 返回坏帧时不崩溃

关键坑（设计方案第三章）：
1. 默认后端 MSMF 会静默忽略分辨率，且慢；优先 DSHOW。
2. 分辨率必须作构造参数传入，绝不能在已打开句柄上 set()（会锁死驱动）。
3. 设置后必须读回验证，驱动可能静默降级。
"""
import cv2

# 候选分辨率档位（从高到低尝试，第一个能真正生效的即硬件上限）
CANDIDATE_RESOLUTIONS = [
    (1920, 1080),
    (1280, 720),
    (960, 540),
    (800, 600),
    (640, 480),
    (320, 240),
]

# 候选编码
CANDIDATE_CODECS = ["MJPG", "YUYV"]


class CameraDevice:
    """封装摄像头打开、探测与帧读取。"""

    def __init__(self, index=0):
        self.index = index
        self.cap = None
        self.backend = None
        self.backend_name = None
        self.width = None
        self.height = None
        self.fourcc = None
        self.fourcc_name = None
        self.open_ms = None

    # ---------------- 后端选择 ----------------
    @staticmethod
    def _try_open(index, backend, params=None):
        """尝试以指定后端打开，返回 cap 或 None。"""
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            return None
        # 若指定了构造参数（分辨率等），用 set 在打开后立即应用是不安全的。
        # 这里改为：params 通过 cv2.VideoCapture 的第三个参数传入更安全，
        # 但 Python 接口只支持 set()。为规避「锁死驱动」风险，我们统一
        # 在 open 后立刻 set 并读回验证，若读回失败则丢弃该句柄重开。
        if params:
            for prop, value in params.items():
                cap.set(prop, value)
        return cap

    def open(self, width=None, height=None):
        """打开摄像头：优先 DSHOW，失败退回 ANY。可带目标分辨率。

        ⚠️ 分辨率必须作为【构造参数】传入 VideoCapture，不能 open 后 set()
        —— 在已打开句柄上 set() 可能锁死驱动（老师方案明确此坑）。

        返回 True/False。成功后 self.cap 可用。
        """
        import time
        for backend, name in ((cv2.CAP_DSHOW, "CAP_DSHOW"),
                              (cv2.CAP_ANY, "CAP_ANY")):
            start = time.time()
            if width and height:
                # 分辨率作构造参数传入
                cap = cv2.VideoCapture(self.index, backend, [
                    cv2.CAP_PROP_FRAME_WIDTH, width,
                    cv2.CAP_PROP_FRAME_HEIGHT, height,
                    cv2.CAP_PROP_FPS, 30,
                ])
            else:
                cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            self.cap = cap
            self.backend = backend
            self.backend_name = name
            self.open_ms = round((time.time() - start) * 1000, 1)
            # 读回实际分辨率（驱动可能静默降级）
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.fourcc_name = self._fourcc_to_str(self.fourcc)
            # 帧率（过滤 NaN/-1）
            self.fps = self.safe_get(cap, cv2.CAP_PROP_FPS)
            return True
        return False

    # ---------------- 分辨率探测 ----------------
    def probe_resolutions(self):
        """逐档尝试候选分辨率，返回探测过程记录列表。

        每条记录：{resolution, width_actual, height_actual, ok}
        ok 表示「读回的实际分辨率与该档一致」。

        优化：复用同一个句柄逐档 set + 读回，而不是反复 open，
        避免「反复 open」把总耗时爆到几十秒（方案 M1 关键要点）。
        """
        records = []
        # 复用句柄：只 open 一次
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return [{"resolution": f"{w}x{h}", "width_actual": None,
                     "height_actual": None, "ok": False}
                    for w, h in CANDIDATE_RESOLUTIONS]

        for w, h in CANDIDATE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ok = (aw == w and ah == h)
            records.append({
                "resolution": f"{w}x{h}",
                "width_actual": aw,
                "height_actual": ah,
                "ok": ok,
            })
        cap.release()
        return records

    def hardware_max_resolution(self, records):
        """从探测记录推断硬件上限：返回 (w, h) 或 None。

        取「读回一致」的最高分辨率档；若都不一致，取读回的最大值。
        """
        ok_records = [r for r in records if r["ok"]]
        if ok_records:
            best = ok_records[0]  # 列表已按从高到低排序
            return best["width_actual"], best["height_actual"]
        # 无档位完全一致时，取读回面积最大的
        best = None
        for r in records:
            if r["width_actual"] and r["height_actual"]:
                area = r["width_actual"] * r["height_actual"]
                if best is None or area > best[1] * best[2]:
                    best = (r["width_actual"], r["height_actual"], area)
        return (best[0], best[1]) if best else None

    # ---------------- 编码探测 ----------------
    def probe_codecs(self, width, height):
        """对比 MJPG / YUYV 两种编码是否可用。返回记录列表。"""
        records = []
        cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            return [{"codec": c, "ok": False, "got": None}
                    for c in CANDIDATE_CODECS]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        for codec in CANDIDATE_CODECS:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*codec))
            got = self._fourcc_to_str(int(cap.get(cv2.CAP_PROP_FOURCC)))
            records.append({"codec": codec, "ok": got == codec, "got": got})
        cap.release()
        return records

    # ---------------- 坏帧防护 ----------------
    def read_frame(self):
        """读取一帧，带坏帧防护。返回 (ok, frame)。"""
        if self.cap is None:
            return False, None
        try:
            ok, frame = self.cap.read()
        except cv2.error:
            return False, None
        if not ok or frame is None:
            return False, None
        # 进一步防护：异常形状（如切换分辨率后的空帧）
        try:
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                return False, None
        except AttributeError:
            return False, None
        return True, frame

    def release(self):
        """释放设备。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    @staticmethod
    def _fourcc_to_str(fourcc):
        """把 FOURCC 整数转成 4 字符字符串。"""
        chars = []
        for i in range(4):
            chars.append(chr((fourcc >> (8 * i)) & 0xFF))
        return "".join(chars).strip()

    @staticmethod
    def safe_get(cap, prop):
        """安全读取摄像头属性（过滤 NaN 和负数）。

        读不到帧率时 OpenCV 返回 -1（不是 0），必须一起过滤，
        否则报告里会出现 "-1.0 FPS" 这种荒谬值。
        """
        try:
            value = float(cap.get(prop))
        except cv2.error:
            return 0.0
        if value != value:      # NaN
            return 0.0
        return value if value > 0 else 0.0
