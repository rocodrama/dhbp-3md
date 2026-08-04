from __future__ import annotations

from dataclasses import dataclass, field
import shlex


@dataclass(frozen=True)
class RemoteVLLMServer:
    host: str
    model: str
    port: int = 8000
    gpu_devices: str | None = None
    vllm_bin: str = "vllm"
    log_path: str = "vllm-3md.log"
    extra_args: tuple[str, ...] = field(default_factory=tuple)

    def start_command(self) -> list[str]:
        args = [
            *shlex.split(self.vllm_bin),
            "serve",
            self.model,
            "--host",
            "0.0.0.0",
            "--port",
            str(self.port),
            *self.extra_args,
        ]
        remote = " ".join(shlex.quote(part) for part in args)
        remote = f"nohup {remote} > {shlex.quote(self.log_path)} 2>&1 &"
        if self.gpu_devices:
            remote = f"CUDA_VISIBLE_DEVICES={shlex.quote(self.gpu_devices)} {remote}"
        return ["ssh", self.host, remote]

    @staticmethod
    def health_url(base_url: str) -> str:
        return f"{base_url.rstrip('/')}/models"
