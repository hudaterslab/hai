from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import socket
from typing import Dict


DEFAULT_PORT = 20000
FRAME_SIZE = 10


class LampState(IntEnum):
    OFF = 0
    ON = 1
    BLINK = 2


@dataclass(frozen=True)
class DeviceStatus:
    sound_group: int
    red: LampState
    yellow: LampState
    green: LampState
    blue: LampState
    white: LampState
    sound_channel: int

    def as_dict(self) -> Dict[str, int | str]:
        return {
            "sound_group": self.sound_group,
            "red": self.red.name,
            "yellow": self.yellow.name,
            "green": self.green.name,
            "blue": self.blue.name,
            "white": self.white.name,
            "sound_channel": self.sound_channel,
        }


class QLightETN:
    """
    TCP client for QLight ETN Ethernet tower lamps.
    """

    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def read_status(self) -> DeviceStatus:
        frame = bytearray(FRAME_SIZE)
        frame[0] = ord("R")
        response = self._send_frame(frame, expect_response=True)
        return self._parse_status(response)

    def write_status(
        self,
        *,
        red: LampState | None = None,
        yellow: LampState | None = None,
        green: LampState | None = None,
        blue: LampState | None = None,
        white: LampState | None = None,
        sound_channel: int | None = None,
        sound_group: int = 0,
        keep_previous: int = 100,
    ) -> None:
        self._validate_sound_group(sound_group)
        frame = bytearray([ord("W"), sound_group, keep_previous, keep_previous, keep_previous, keep_previous, keep_previous, 0, 0, 0])

        if red is not None:
            frame[2] = int(red)
        if yellow is not None:
            frame[3] = int(yellow)
        if green is not None:
            frame[4] = int(green)
        if blue is not None:
            frame[5] = int(blue)
        if white is not None:
            frame[6] = int(white)
        if sound_channel is not None:
            self._validate_sound_channel(sound_channel)
            frame[7] = sound_channel
        else:
            frame[7] = keep_previous

        self._send_frame(frame, expect_response=False)

    def all_off(self) -> None:
        self.write_status(
            red=LampState.OFF,
            yellow=LampState.OFF,
            green=LampState.OFF,
            blue=LampState.OFF,
            white=LampState.OFF,
            sound_channel=0,
        )

    def _send_frame(self, frame: bytes, *, expect_response: bool) -> bytes:
        if len(frame) != FRAME_SIZE:
            raise ValueError(f"frame must be {FRAME_SIZE} bytes")

        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.settimeout(self.timeout)
            sock.sendall(frame)

            if not expect_response:
                return b""

            response = self._recv_exact(sock, FRAME_SIZE)
            if response[0] != ord("A"):
                raise RuntimeError(f"unexpected response header: {response[0]!r}")
            return response

    @staticmethod
    def _recv_exact(sock: socket.socket, size: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < size:
            chunk = sock.recv(size - len(chunks))
            if not chunk:
                raise RuntimeError("connection closed before full response was received")
            chunks.extend(chunk)
        return bytes(chunks)

    @staticmethod
    def _parse_status(response: bytes) -> DeviceStatus:
        if len(response) != FRAME_SIZE:
            raise ValueError(f"response must be {FRAME_SIZE} bytes")

        return DeviceStatus(
            sound_group=response[1],
            red=LampState(response[2]),
            yellow=LampState(response[3]),
            green=LampState(response[4]),
            blue=LampState(response[5]),
            white=LampState(response[6]),
            sound_channel=response[7],
        )

    @staticmethod
    def _validate_sound_group(sound_group: int) -> None:
        if not 0 <= sound_group <= 4:
            raise ValueError("sound_group must be between 0 and 4")

    @staticmethod
    def _validate_sound_channel(sound_channel: int) -> None:
        if not 0 <= sound_channel <= 5:
            raise ValueError("sound_channel must be between 0 and 5")
