from typing import List

from src.scenes.scene import Scene


class SceneStack:
    def __init__(self):
        self.scenes: List[Scene] = []

    def __len__(self):
        return len(self.scenes)

    def push(self, scene):
        self.scenes.append(scene)
        return self

    def pop(self):
        self.scenes.pop().exit_scene()
        return self

    def top(self):
        return self.scenes[-1]

    def is_empty(self):
        return len(self.scenes) == 0

    def clear(self):
        self.scenes.clear()
        return self

