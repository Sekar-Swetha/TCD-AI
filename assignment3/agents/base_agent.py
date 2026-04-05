from abc import ABC, abstractmethod


class BaseAgent(ABC):

    @abstractmethod
    def select_action(self, game, legal_moves=None):
        pass
