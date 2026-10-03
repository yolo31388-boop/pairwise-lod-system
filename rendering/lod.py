"""LOD细节层次系统"""
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mesh:
    name: str
    vertex_count: int
    triangle_count: int
    material_params: dict = field(default_factory=dict)


@dataclass
class RenderObject:
    id: int
    position: tuple[float, float, float]
    meshes: list[Mesh] = field(default_factory=list)
    current_lod: int = 0
    importance: float = 1.0
    bounding_radius: float = 1.0
    material: Optional[str] = None
    material_params: dict = field(default_factory=dict)
    transparent: bool = False
    animation_time: float = 0.0
    animation_speed: float = 1.0


@dataclass
class LODTransition:
    """LOD切换过渡：交叉淡化，避免直接pop"""
    from_lod: int
    to_lod: int
    duration: float
    progress: float = 0.0

    @property
    def alpha(self) -> float:
        if self.duration <= 0:
            return 1.0
        return min(1.0, self.progress / self.duration)

    @property
    def done(self) -> bool:
        return self.progress >= self.duration


@dataclass
class RenderBatch:
    """渲染批次：同模型同材质的物体实例化合并为一次Draw Call"""
    mesh_name: str
    material: Optional[str]
    instances: list = field(default_factory=list)
    transparent: bool = False

    @property
    def instance_count(self) -> int:
        return len(self.instances)


class LODSystem:
    def __init__(
        self,
        camera_position: tuple[float, float, float] = (0, 0, 0),
        camera_forward: tuple[float, float, float] = (0, 0, 1),
        screen_width: int = 1920,
        screen_height: int = 1080,
        fov_y: float = 60.0,
        hysteresis: float = 0.1,
        transition_duration: float = 0.3,
        streaming_radius: float = 100.0,
        preload_radius: float = 150.0,
        unload_radius: float = 200.0,
        max_loads_per_frame: int = 8,
    ):
        self.camera_position = camera_position
        self.camera_forward = camera_forward
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.fov_y = fov_y
        self.hysteresis = hysteresis
        self.transition_duration = transition_duration
        self.streaming_radius = streaming_radius
        self.preload_radius = preload_radius
        self.unload_radius = unload_radius
        self.max_loads_per_frame = max_loads_per_frame
        self.objects: dict[int, RenderObject] = {}
        self.lod_thresholds = [10.0, 50.0, 100.0]  # 参考物体(半径1)的距离阈值
        self.reference_radius = 1.0
        self.reference_screen_height = 1080  # 距离阈值换算像素阈值的参考分辨率
        self.loaded_assets: set[int] = set()
        self._prev_lod: dict[int, int] = {}
        self._transitions: dict[int, LODTransition] = {}

    def add_object(self, obj: RenderObject):
        self.objects[obj.id] = obj

    def set_camera(self, position: tuple[float, float, float],
                   forward: Optional[tuple[float, float, float]] = None):
        self.camera_position = position
        if forward is not None:
            self.camera_forward = forward

    # ---------- LOD计算：屏幕占比 + 重要性 + 滞后 ----------

    def calculate_lod_level(self, obj: RenderObject) -> int:
        """计算LOD等级：基于屏幕占比(像素)，考虑重要性与相机方向，并带滞后"""
        max_lod = max(0, len(obj.meshes) - 1) if obj.meshes else len(self.lod_thresholds)
        # 相机方向：背后的物体直接用最低LOD
        if self._is_behind_camera(obj):
            return max_lod
        # 屏幕占比换算成像素：大屏幕同样占比覆盖更多像素 -> 用更高LOD
        coverage = self.calculate_screen_coverage(obj)
        pixels = coverage * self.screen_height
        # 重要性：主角等效像素放大，背景缩小
        effective = pixels * max(obj.importance, 1e-6)
        thresholds = self._pixel_thresholds()
        raw = 0
        while raw < len(thresholds) and effective < thresholds[raw]:
            raw += 1
        raw = min(raw, max_lod)
        return self._apply_hysteresis(obj, raw, effective, thresholds, max_lod)

    def _apply_hysteresis(self, obj: RenderObject, raw_lod: int, effective: float,
                          thresholds: list[float], max_lod: int) -> int:
        """滞后：升质需明显高于阈值，降质需明显低于阈值，防止阈值附近闪烁"""
        current = min(obj.current_lod, max_lod)
        margin = self.hysteresis
        lod = current
        if raw_lod > current:
            while lod < max_lod and effective < thresholds[lod] * (1.0 - margin):
                lod += 1
        elif raw_lod < current:
            while lod > 0 and effective > thresholds[lod - 1] * (1.0 + margin):
                lod -= 1
        return lod

    def _pixel_thresholds(self) -> list[float]:
        """把参考物体的距离阈值按参考分辨率换算为固定像素阈值"""
        tan_half_fov = math.tan(math.radians(self.fov_y) / 2.0)
        return [
            self.reference_radius / (t * tan_half_fov) * self.reference_screen_height
            for t in self.lod_thresholds
        ]

    def calculate_screen_coverage(self, obj: RenderObject,
                                  screen_width: Optional[int] = None) -> float:
        """计算屏幕占比：由包围球半径、距离和视场角推算的屏幕高度占比"""
        dist = self._center_distance(obj)
        if dist <= 1e-6:
            return 1.0
        tan_half_fov = math.tan(math.radians(self.fov_y) / 2.0)
        coverage = obj.bounding_radius / (dist * tan_half_fov)
        return max(0.0, min(1.0, coverage))

    # ---------- 距离：包围盒 + 平方优化 + 相机方向 ----------

    def calculate_distance(self, obj: RenderObject) -> float:
        """包围盒感知距离：相机到包围球表面的距离，大物体距离更近"""
        return max(0.0, self._center_distance(obj) - obj.bounding_radius)

    def _calculate_distance(self, obj: RenderObject) -> float:
        return self.calculate_distance(obj)

    def _distance_sq(self, obj: RenderObject) -> float:
        """平方距离：只做比较时避免开平方"""
        dx = obj.position[0] - self.camera_position[0]
        dy = obj.position[1] - self.camera_position[1]
        dz = obj.position[2] - self.camera_position[2]
        return dx * dx + dy * dy + dz * dz

    def _center_distance(self, obj: RenderObject) -> float:
        return math.sqrt(self._distance_sq(obj))

    def _is_behind_camera(self, obj: RenderObject) -> bool:
        dx = obj.position[0] - self.camera_position[0]
        dy = obj.position[1] - self.camera_position[1]
        dz = obj.position[2] - self.camera_position[2]
        fx, fy, fz = self.camera_forward
        return dx * fx + dy * fy + dz * fz < 0.0

    # ---------- 模型切换：过渡 + 状态同步 + 材质同步 ----------

    def switch_lod_model(self, obj: RenderObject, new_lod: int) -> bool:
        """切换LOD模型：交叉淡化过渡，动画状态与材质参数同步保留"""
        if new_lod < 0 or new_lod >= len(obj.meshes):
            return False
        if new_lod == obj.current_lod:
            return True
        old_lod = obj.current_lod
        # 状态同步：保存动画状态，切换后不重置
        animation_time = obj.animation_time
        animation_speed = obj.animation_speed
        # 材质同步：材质参数随切换保留到新模型
        material_params = dict(obj.material_params)
        obj.current_lod = new_lod
        obj.animation_time = animation_time
        obj.animation_speed = animation_speed
        obj.material_params.update(material_params)
        self._sync_material_to_mesh(obj)
        # 过渡：记录交叉淡化状态而不是直接pop
        self._transitions[obj.id] = LODTransition(
            from_lod=old_lod, to_lod=new_lod, duration=self.transition_duration,
        )
        return True

    def _sync_material_to_mesh(self, obj: RenderObject):
        if obj.meshes:
            obj.meshes[obj.current_lod].material_params.update(obj.material_params)

    def get_transition(self, obj_id: int) -> Optional[LODTransition]:
        return self._transitions.get(obj_id)

    def get_render_lods(self, obj: RenderObject) -> list[tuple[int, float]]:
        """过渡期间返回两个LOD及混合权重，用于交叉淡化渲染"""
        transition = self._transitions.get(obj.id)
        if transition is None or transition.done:
            return [(obj.current_lod, 1.0)]
        return [
            (transition.from_lod, 1.0 - transition.alpha),
            (transition.to_lod, transition.alpha),
        ]

    def update(self, dt: float = 0.0):
        """更新所有对象的LOD（带滞后），并推进切换过渡"""
        for obj in self.objects.values():
            new_lod = self.calculate_lod_level(obj)
            if new_lod != obj.current_lod:
                self._prev_lod[obj.id] = obj.current_lod
                self.switch_lod_model(obj, new_lod)
        self._update_transitions(dt)

    def _update_transitions(self, dt: float):
        finished = []
        for obj_id, transition in self._transitions.items():
            transition.progress += dt
            if transition.done:
                finished.append(obj_id)
        for obj_id in finished:
            del self._transitions[obj_id]

    # ---------- 批处理：按材质 + 实例化 + 排序 ----------

    def batch_render(self, objects: Optional[list] = None) -> list:
        """合批渲染：不透明在前透明在后，同模型同材质实例化为一次Draw Call"""
        objs = list(self.objects.values()) if objects is None else list(objects)
        # 排序：不透明先渲染（前到后），透明最后渲染（后到前）
        opaque = sorted((o for o in objs if not o.transparent), key=self._distance_sq)
        transparent = sorted((o for o in objs if o.transparent),
                             key=self._distance_sq, reverse=True)
        batches: list[RenderBatch] = []
        index: dict[tuple, RenderBatch] = {}
        for obj in opaque + transparent:
            mesh = obj.meshes[obj.current_lod] if obj.meshes else None
            # 按模型+材质分组合批：相同模型不同材质不能合批
            key = (mesh.name if mesh else None, obj.material, obj.transparent)
            batch = index.get(key)
            if batch is None:
                batch = RenderBatch(
                    mesh_name=key[0] or "",
                    material=obj.material,
                    transparent=obj.transparent,
                )
                index[key] = batch
                batches.append(batch)
            # 实例化：同批物体合并为一次Draw Call
            batch.instances.append(obj)
        return batches

    def get_draw_call_count(self, objects: Optional[list] = None) -> int:
        return len(self.batch_render(objects))

    # ---------- 流送：卸载 + 预加载 + 优先级 ----------

    def stream_assets(self, dt: float = 0.0) -> dict:
        """资源流送：按优先级加载/预加载进入范围的资源，卸载远离的资源"""
        stream_sq = self.streaming_radius ** 2
        preload_sq = self.preload_radius ** 2
        unload_sq = self.unload_radius ** 2
        candidates: list[tuple[float, int, bool]] = []  # (优先级, id, 是否预加载)
        unloaded: list[int] = []
        for obj in self.objects.values():
            dist_sq = self._distance_sq(obj)  # 平方距离比较，避免开平方
            if dist_sq <= preload_sq:
                if obj.id not in self.loaded_assets:
                    # 优先级：重要性越高、距离越近越先加载，主角先于背景
                    priority = obj.importance / (1.0 + math.sqrt(dist_sq))
                    candidates.append((priority, obj.id, dist_sq > stream_sq))
            elif dist_sq > unload_sq and obj.id in self.loaded_assets:
                # 卸载：远离的物体释放显存
                self.loaded_assets.discard(obj.id)
                unloaded.append(obj.id)
        candidates.sort(key=lambda c: c[0], reverse=True)
        loaded: list[int] = []
        preloaded: list[int] = []
        pending: list[int] = []
        budget = self.max_loads_per_frame
        for _priority, obj_id, is_preload in candidates:
            if budget <= 0:
                pending.append(obj_id)
                continue
            budget -= 1
            self.loaded_assets.add(obj_id)
            (preloaded if is_preload else loaded).append(obj_id)
        return {
            "loaded": loaded,
            "preloaded": preloaded,
            "unloaded": unloaded,
            "pending": pending,
        }

    def is_asset_loaded(self, obj_id: int) -> bool:
        return obj_id in self.loaded_assets

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
