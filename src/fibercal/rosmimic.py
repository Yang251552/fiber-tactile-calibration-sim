"""A ~50-line ROS2-style pub/sub layer.

This is NOT ROS2. It mirrors the node/topic/message concepts so the data flow
is explicit and ROS-shaped, without the colcon/DDS install friction. A real
rclpy port of any node is a drop-in next step (see docs/roadmap.md).

Topic graph of this project:

    IndenterControllerNode --/cmd_force-->  SimNode --/contact_state--> SensorNode
                                                                            |
                                                                   /sensor_readings
                                                                            v
                              EstimatorNode  <--/dataset--  LoggerNode
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable


class Bus:
    """In-process message bus standing in for the ROS2 DDS layer."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[[Any], None]]] = defaultdict(list)
        self.topics: set[str] = set()

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        self._subs[topic].append(callback)
        self.topics.add(topic)

    def publish(self, topic: str, msg: Any) -> None:
        self.topics.add(topic)
        for cb in self._subs[topic]:
            cb(msg)


class Node:
    """Base node. Subclasses publish/subscribe on construction."""

    def __init__(self, bus: Bus, name: str) -> None:
        self.bus = bus
        self.name = name

    def publish(self, topic: str, msg: Any) -> None:
        self.bus.publish(topic, msg)

    def subscribe(self, topic: str, callback: Callable[[Any], None]) -> None:
        self.bus.subscribe(topic, callback)


def topic_graph(bus: Bus) -> str:
    return "active topics: " + ", ".join(sorted(bus.topics))
