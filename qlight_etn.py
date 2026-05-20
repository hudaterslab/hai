from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import socket
from typing import Dict
import urllib.request


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
    Client for QLight ETN Ethernet tower lamps.

    This device exposes TCP status reads on port 20000, but lamp/sound writes
    are sent through the same HTTP endpoints used by the built-in web UI.
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

        self._send_http_command(f"L?T={sound_group}")

        lamp_commands = (
            (1, red),
            (2, yellow),
            (3, green),
            (4, blue),
            (5, white),
        )
        for lamp_number, state in lamp_commands:
            if state is not None:
                self._send_http_command(f"L?{lamp_number}={self._to_http_lamp_state(state)}")

        if sound_channel is not None:
            self._validate_sound_channel(sound_channel)
            self._send_http_command(f"L?S={sound_channel}")

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

    def _send_http_command(self, command: str) -> str:
        base_url = self.host
        if not base_url.startswith(("http://", "https://")):
            base_url = f"http://{base_url}"
        url = f"{base_url.rstrip('/')}/{command}"

        with urllib.request.urlopen(url, timeout=self.timeout) as response:
            body = response.read().decode("utf-8", errors="replace").strip()

        if body and body != "OK":
            raise RuntimeError(f"unexpected HTTP response for {command}: {body!r}")
        return body

    @staticmethod
    def _to_http_lamp_state(state: LampState | int) -> int:
        state = LampState(state)
        if state == LampState.OFF:
            return 3
        return int(state)

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
        if not 0 <= sound_group <= 5:
            raise ValueError("sound_group must be between 0 and 5")

    @staticmethod
    def _validate_sound_channel(sound_channel: int) -> None:
        if not 0 <= sound_channel <= 5:
            raise ValueError("sound_channel must be between 0 and 5")
