"""LOD细节层次系统"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Material:
    """材质"""
    name: str
    transparent: bool = False
    params: dict = field(default_factory=dict)


@dataclass
class Mesh:
    name: str
    vertex_count: int
    triangle_count: int
    material: Optional[Material] = None
    animation_time: float = 0.0


@dataclass
class RenderObject:
    id: int
    position: tuple[float, float, float]
    meshes: list[Mesh] = field(default_factory=list)  # [high, medium, low]
    current_lod: int = 0
    importance: float = 1.0
    bounding_radius: float = 1.0
    material: Optional[Material] = None
    animation_time: float = 0.0
    loaded: bool = False
    transition: Optional[dict] = None


class LODSystem:
    def __init__(
        self,
        camera_position: tuple[float, float, float] = (0, 0, 0),
        camera_direction: tuple[float, float, float] = (0, 0, 1),
        screen_height: int = 1080,
        fov: float = 60.0,
        view_distance: float = 100.0,
        preload_distance: float = 150.0,
        unload_distance: float = 200.0,
    ):
        self.camera_position = camera_position
        norm = math.sqrt(sum(c * c for c in camera_direction)) or 1.0
        self.camera_direction = tuple(c / norm for c in camera_direction)
        self.screen_height = screen_height
        self.fov = fov
        self.view_distance = view_distance
        self.preload_distance = preload_distance
        self.unload_distance = unload_distance
        self.objects: dict[int, RenderObject] = {}
        self.lod_thresholds = [10.0, 50.0, 100.0]  # high, medium, low
        # 屏幕占比阈值（占屏幕比例，与分辨率无关，大小屏幕通用）
        self.coverage_thresholds = [0.1, 0.02]
        self.hysteresis = 0.1  # 滞后带，防止阈值附近闪烁
        self.behind_penalty = 4.0  # 相机背后物体的覆盖率惩罚
        self.transition_duration = 0.5  # LOD切换过渡时长（秒）
        self.loaded_assets: set[int] = set()
        self._prev_lod: dict[int, int] = {}

    def add_object(self, obj: RenderObject):
        self.objects[obj.id] = obj

    # ---------- LOD计算：屏幕占比 + 重要性 + 滞后 ----------

    def calculate_lod_level(self, obj: RenderObject) -> int:
        """计算LOD等级：基于屏幕占比和重要性，并带滞后防止闪烁"""
        coverage = self._effective_coverage(obj)
        max_lod = max(0, len(obj.meshes) - 1) if obj.meshes else 0
        lod = min(obj.current_lod, max_lod)
        # 降级（更远）：覆盖率需低于阈值*(1-滞后)才切换
        while (
            lod < max_lod
            and lod < len(self.coverage_thresholds)
            and coverage < self.coverage_thresholds[lod] * (1.0 - self.hysteresis)
        ):
            lod += 1
        # 升级（更近）：覆盖率需高于阈值*(1+滞后)才切换
        while lod > 0 and coverage > self.coverage_thresholds[lod - 1] * (1.0 + self.hysteresis):
            lod -= 1
        return lod

    def _effective_coverage(self, obj: RenderObject) -> float:
        """有效屏幕占比：考虑重要性和相机方向"""
        coverage = self.calculate_screen_coverage(obj)
        coverage *= max(obj.importance, 1e-6)
        if self.is_behind_camera(obj):
            coverage /= self.behind_penalty
        return coverage

    def calculate_screen_coverage(self, obj: RenderObject, screen_width: int = 1920) -> float:
        """计算屏幕占比：物体包围球角直径占屏幕的比例（与分辨率无关）"""
        dist = self._center_distance(obj)
        if dist <= 1e-6:
            return 1.0
        return min(1.0, obj.bounding_radius / dist)

    # ---------- 距离：包围盒 + 平方优化 + 相机方向 ----------

    def calculate_distance(self, obj: RenderObject) -> float:
        """计算距离：到包围盒表面的距离，而非中心点"""
        return math.sqrt(self.calculate_distance_squared(obj))

    def calculate_distance_squared(self, obj: RenderObject) -> float:
        """平方距离（到包围盒）：避免开平方，用于比较和排序"""
        center_sq = self._center_distance_squared(obj)
        if obj.bounding_radius <= 0:
            return center_sq
        center = math.sqrt(center_sq)
        surface = max(0.0, center - obj.bounding_radius)
        return surface * surface

    def _center_distance(self, obj: RenderObject) -> float:
        return math.sqrt(self._center_distance_squared(obj))

    def _center_distance_squared(self, obj: RenderObject) -> float:
        dx = obj.position[0] - self.camera_position[0]
        dy = obj.position[1] - self.camera_position[1]
        dz = obj.position[2] - self.camera_position[2]
        return dx * dx + dy * dy + dz * dz

    # 兼容旧的私有接口
    _calculate_distance = calculate_distance

    def is_behind_camera(self, obj: RenderObject) -> bool:
        """判断物体是否在相机背后"""
        dx = obj.position[0] - self.camera_position[0]
        dy = obj.position[1] - self.camera_position[1]
        dz = obj.position[2] - self.camera_position[2]
        dot = (
            dx * self.camera_direction[0]
            + dy * self.camera_direction[1]
            + dz * self.camera_direction[2]
        )
        return dot < 0

    # ---------- 模型切换：过渡 + 状态同步 + 材质同步 ----------

    def switch_lod_model(self, obj: RenderObject, new_lod: int) -> bool:
        """切换LOD模型：平滑过渡，并同步动画状态和材质参数"""
        if new_lod < 0 or new_lod >= len(obj.meshes):
            return False
        old_lod = obj.current_lod
        if new_lod == old_lod:
            return True
        old_mesh = obj.meshes[old_lod] if 0 <= old_lod < len(obj.meshes) else None
        new_mesh = obj.meshes[new_lod]
        self._sync_animation_state(obj, old_mesh, new_mesh)
        self._sync_materials(obj, old_mesh, new_mesh)
        # 交叉淡化过渡，避免直接pop
        obj.transition = {
            "from_lod": old_lod,
            "to_lod": new_lod,
            "progress": 0.0,
            "duration": self.transition_duration,
        }
        obj.current_lod = new_lod
        self._prev_lod[obj.id] = old_lod
        return True

    def _sync_animation_state(self, obj: RenderObject, old_mesh: Optional[Mesh], new_mesh: Mesh):
        """动画状态同步：切换模型后动画从原进度继续播放，不重置"""
        source_time = obj.animation_time
        if old_mesh is not None and old_mesh.animation_time:
            source_time = old_mesh.animation_time
        new_mesh.animation_time = source_time
        obj.animation_time = source_time

    def _sync_materials(self, obj: RenderObject, old_mesh: Optional[Mesh], new_mesh: Mesh):
        """材质同步：高模的材质参数传递到低模，避免丢失"""
        source = None
        if old_mesh is not None and old_mesh.material is not None:
            source = old_mesh.material
        elif obj.material is not None:
            source = obj.material
        if source is None:
            return
        if new_mesh.material is None:
            new_mesh.material = Material(
                name=source.name,
                transparent=source.transparent,
                params=dict(source.params),
            )
        else:
            for key, value in source.params.items():
                new_mesh.material.params.setdefault(key, value)

    def update(self, dt: float = 0.1):
        """更新所有对象的LOD（带滞后），并推进切换过渡"""
        for obj_id, obj in self.objects.items():
            new_lod = self.calculate_lod_level(obj)
            if new_lod != obj.current_lod:
                self.switch_lod_model(obj, new_lod)
            self._prev_lod[obj_id] = obj.current_lod
            if obj.transition is not None:
                duration = max(obj.transition["duration"], 1e-6)
                obj.transition["progress"] += dt / duration
                if obj.transition["progress"] >= 1.0:
                    obj.transition = None

    # ---------- 批处理：按材质 + 实例化 + 排序 ----------

    def batch_render(self, objects: Optional[list[RenderObject]] = None) -> list[dict]:
        """批处理：按(模型,材质)分组实例化合批，不透明在前透明在后"""
        objs = list(objects) if objects is not None else list(self.objects.values())
        groups: dict[tuple, dict] = {}
        for obj in objs:
            if not obj.meshes:
                continue
            lod = min(obj.current_lod, len(obj.meshes) - 1)
            mesh = obj.meshes[lod]
            material = obj.material if obj.material is not None else mesh.material
            key = (
                mesh.name,
                mesh.vertex_count,
                mesh.triangle_count,
                material.name if material is not None else None,
            )
            batch = groups.get(key)
            if batch is None:
                batch = {
                    "mesh": mesh,
                    "material": material,
                    "instances": [],
                    "instance_count": 0,
                    "transparent": material.transparent if material is not None else False,
                    "_dist_sq": float("inf"),
                }
                groups[key] = batch
            batch["instances"].append(obj)
            batch["instance_count"] += 1
            # 平方距离排序键，避免开平方
            batch["_dist_sq"] = min(batch["_dist_sq"], self._center_distance_squared(obj))
        opaque = [b for b in groups.values() if not b["transparent"]]
        transparent = [b for b in groups.values() if b["transparent"]]
        # 不透明从前到后，透明从后到前
        opaque.sort(key=lambda b: b["_dist_sq"])
        transparent.sort(key=lambda b: -b["_dist_sq"])
        for batch in opaque + transparent:
            del batch["_dist_sq"]
        return opaque + transparent

    # ---------- 流送：卸载 + 预加载 + 优先级 ----------

    def stream_assets(self) -> dict:
        """资源流送：按优先级加载/预加载进入范围的资源，卸载远离的资源"""
        stats = {"loaded": [], "unloaded": [], "preloaded": []}
        to_load: list[tuple[RenderObject, float]] = []
        for obj in self.objects.values():
            dist = self.calculate_distance(obj)
            if dist > self.unload_distance:
                # 卸载：不用的资源释放显存
                if obj.loaded:
                    obj.loaded = False
                    self.loaded_assets.discard(obj.id)
                    stats["unloaded"].append(obj.id)
            elif dist <= self.preload_distance:
                if not obj.loaded:
                    to_load.append((obj, dist))
        # 优先级：重要性高的（主角）先加载，同等重要性近的先加载
        to_load.sort(key=lambda item: (-item[0].importance, item[1]))
        for obj, dist in to_load:
            obj.loaded = True
            self.loaded_assets.add(obj.id)
            if dist <= self.view_distance:
                stats["loaded"].append(obj.id)
            else:
                # 预加载：进入视野前提前加载，避免卡顿
                stats["preloaded"].append(obj.id)
        return stats

    # ---------- 统计 ----------

    def get_render_stats(self) -> dict:
        """获取渲染统计"""
        stats = {"high": 0, "medium": 0, "low": 0}
        for obj in self.objects.values():
            if obj.current_lod == 0:
                stats["high"] += 1
            elif obj.current_lod == 1:
                stats["medium"] += 1
            else:
                stats["low"] += 1
        return stats
