"""LOD细节层次系统 - 存在bug"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Mesh:
    name: str
    vertex_count: int
    triangle_count: int


@dataclass
class RenderObject:
    id: int
    position: tuple[float, float, float]
    meshes: list[Mesh] = field(default_factory=list)  # [high, medium, low]
    current_lod: int = 0
    importance: float = 1.0


class LODSystem:
    def __init__(self, camera_position: tuple[float, float, float] = (0, 0, 0)):
        self.camera_position = camera_position
        self.objects: dict[int, RenderObject] = {}
        self.lod_thresholds = [10.0, 50.0, 100.0]  # high, medium, low
        self._prev_lod: dict[int, int] = {}

    def add_object(self, obj: RenderObject):
        self.objects[obj.id] = obj

    def calculate_lod_level(self, obj: RenderObject) -> int:
        """计算LOD等级 - BUG: 只看距离不看屏幕占比和重要性"""
        dist = self._calculate_distance(obj)
        # BUG: 只看距离，不看屏幕占比和重要性
        if dist < self.lod_thresholds[0]:
            return 0
        elif dist < self.lod_thresholds[1]:
            return 1
        else:
            return 2

    def _calculate_distance(self, obj: RenderObject) -> float:
        """计算距离 - BUG: 只算中心点不算包围盒"""
        # BUG: 只算中心点，不算包围盒
        dx = obj.position[0] - self.camera_position[0]
        dy = obj.position[1] - self.camera_position[1]
        dz = obj.position[2] - self.camera_position[2]
        return (dx * dx + dy * dy + dz * dz) ** 0.5

    def switch_lod_model(self, obj: RenderObject, new_lod: int) -> bool:
        """切换LOD模型 - BUG: 不做过渡，直接切换"""
        # BUG: 不做过渡，直接切换
        if new_lod < 0 or new_lod >= len(obj.meshes):
            return False
        obj.current_lod = new_lod
        return True

    def update(self):
        """更新所有对象的LOD - BUG: 不做滞后，阈值附近闪烁"""
        for obj_id, obj in self.objects.items():
            new_lod = self.calculate_lod_level(obj)
            # BUG: 不做滞后，直接切换
            if new_lod != obj.current_lod:
                self.switch_lod_model(obj, new_lod)

    def calculate_screen_coverage(self, obj: RenderObject, screen_width: int = 1920) -> float:
        """计算屏幕占比 - BUG: 实现不正确"""
        # BUG: 简单估算，不准确
        dist = self._calculate_distance(obj)
        if dist == 0:
            return 1.0
        # 假设物体大小为1
        return min(1.0, 1.0 / dist)

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
