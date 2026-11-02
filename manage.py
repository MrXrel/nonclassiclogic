"""Start/stop the local macOS service independently of the caller's lifetime."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import re
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
LABEL = "local.rayleighlab"
URL = "http://127.0.0.1:8501"


def domain():
    return f"gui/{os.getuid()}"


def target():
    return f"{domain()}/{LABEL}"


def healthy():
    try:
        with urlopen(URL + "/_stcore/health", timeout=1) as response:
            return response.status == 200 and response.read().strip() == b"ok"
    except (OSError, TimeoutError):
        return False


def service_info():
    result = subprocess.run(["/bin/launchctl", "print", target()], capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else None


def owned(info):
    # Refuse to replace a service installed from another project directory.
    return str(ROOT / "app.py") in info


def service_pid(info):
    match = re.search(r"\bpid = (\d+)", info or "")
    return int(match.group(1)) if match else None


def create_service_spec():
    python = ROOT / ".venv" / "bin" / "python"
    if not python.exists():
        raise RuntimeError("Сначала запустите run.command для установки зависимостей.")
    return {
        "Label": LABEL,
        "ProgramArguments": [str(python), "-m", "streamlit", "run", str(ROOT / "app.py"),
                             "--server.address", "127.0.0.1", "--server.port", "8501",
                             "--server.headless", "true"],
        "WorkingDirectory": str(ROOT),
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 5,
        "ProcessType": "Background",
        "StandardOutPath": str(RUNTIME / "server.stdout.log"),
        "StandardErrorPath": str(RUNTIME / "server.stderr.log"),
        "EnvironmentVariables": {"PATH": str(ROOT / ".venv" / "bin") + ":/usr/bin:/bin:/usr/sbin:/sbin"},
    }


def start(open_browser=True):
    info = service_info()
    if info and not owned(info):
        raise RuntimeError("Сервис RayleighLab уже зарегистрирован из другой папки. Остановите его из той папки.")
    if info and healthy():
        print(f"Приложение уже работает. PID: {service_pid(info)}\n{URL}")
        if open_browser:
            webbrowser.open(URL)
        return
    if not info:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", 8501)) == 0:
                raise RuntimeError("Порт 8501 занят другим процессом. Остановите прежний сервер и повторите запуск.")
        RUNTIME.mkdir(exist_ok=True)
        plist = RUNTIME / (LABEL + ".plist")
        with plist.open("wb") as stream:
            plistlib.dump(create_service_spec(), stream)
        result = subprocess.run(["/bin/launchctl", "bootstrap", domain(), str(plist)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Не удалось зарегистрировать фоновый процесс: " + result.stderr.strip())
    else:
        result = subprocess.run(["/bin/launchctl", "kickstart", target()], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("Не удалось запустить сервис: " + result.stderr.strip())
    deadline = time.monotonic() + 25
    while time.monotonic() < deadline:
        if healthy():
            info = service_info()
            print(f"RayleighLab запущен независимо. PID: {service_pid(info)}\n{URL}")
            if open_browser:
                webbrowser.open(URL)
            return
        time.sleep(.2)
    raise RuntimeError("Сервер не ответил за 25 секунд. Диагностика: runtime/server.stderr.log")


def stop():
    info = service_info()
    if not info:
        print("Сервис уже остановлен.")
        return
    if not owned(info):
        raise RuntimeError("Этот сервис принадлежит проекту в другой папке; используйте его stop.command.")
    result = subprocess.run(["/bin/launchctl", "bootout", target()], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError("Не удалось остановить сервис: " + result.stderr.strip())
    # bootout can return while launchd is still removing the job.
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if service_info() is None and not healthy():
            break
        time.sleep(.1)
    else:
        raise RuntimeError("Сервис ещё завершает работу. Проверьте статус перед повторным запуском.")
    print("RayleighLab остановлен.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "status"], nargs="?", default="start")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin":
        parser.error("Фоновый сервис настроен для macOS. На других ОС: python -m streamlit run app.py")
    try:
        if args.action == "start":
            start(open_browser=not args.no_browser)
        elif args.action == "stop":
            stop()
        else:
            info = service_info()
            if info and owned(info) and healthy():
                print(f"Работает. PID: {service_pid(info)}\n{URL}")
            else:
                print("Сервис не работает.")
                return 1
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
