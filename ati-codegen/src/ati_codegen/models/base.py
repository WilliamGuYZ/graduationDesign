from __future__ import annotations

from abc import ABC, abstractmethod

from ..prompts import PromptBundle
from ..types import Generation, Problem


class CodeGenModel(ABC):
    @abstractmethod
    def generate(self, problem: Problem, prompt: PromptBundle) -> Generation:
        raise NotImplementedError

