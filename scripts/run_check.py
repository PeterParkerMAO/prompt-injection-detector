"""Script de verificação de qualidade e segurança (executado pelo alvo 'make check').

Executa ruff, pytest e, se instalados no ambiente, bandit e pip-audit.
Grava a saída consolidada em reports/check.txt (sem conter dados nem segredos).
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
import sys
from pathlib import Path


def run_tool(name: str, cmd: list[str]) -> tuple[int, str]:
    print(f"--> Executando {name}...", flush=True)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        output = proc.stdout + "\n" + proc.stderr
        return proc.returncode, output.strip()
    except Exception as exc:
        return 1, f"Erro ao executar {name}: {exc}"


def main() -> None:
    lines = [
        "===========================================================",
        f"Verificação de Integridade e Qualidade ({dt.datetime.now(dt.UTC).isoformat()})",
        "===========================================================",
        "",
    ]
    all_passed = True

    # 1. Ruff (linter)
    code, out = run_tool("ruff", ["uv", "run", "ruff", "check", "."])
    lines.append(f"[RUFF] Status: {'PASSED' if code == 0 else 'FAILED'}")
    lines.append(out)
    lines.append("-" * 50)
    if code != 0:
        all_passed = False

    # 2. Pytest (testes unitários)
    code, out = run_tool("pytest", ["uv", "run", "pytest", "-q"])
    lines.append(f"[PYTEST] Status: {'PASSED' if code == 0 else 'FAILED'}")
    lines.append(out)
    lines.append("-" * 50)
    if code != 0:
        all_passed = False

    # 3. Bandit (se instalado)
    if shutil.which("bandit") or subprocess.run(["uv", "run", "python", "-c", "import bandit"], capture_output=True).returncode == 0:
        code, out = run_tool("bandit", ["uv", "run", "bandit", "-r", "src/"])
        lines.append(f"[BANDIT] Status: {'PASSED' if code == 0 else 'FAILED'}")
        lines.append(out)
        lines.append("-" * 50)
        if code != 0:
            all_passed = False
    else:
        lines.append("[BANDIT] Não instalado no ambiente (ignorado)")
        lines.append("-" * 50)

    # 4. Pip-audit (se instalado)
    if shutil.which("pip-audit") or subprocess.run(["uv", "run", "python", "-c", "import pip_audit"], capture_output=True).returncode == 0:
        code, out = run_tool("pip-audit", ["uv", "run", "pip-audit"])
        lines.append(f"[PIP-AUDIT] Status: {'PASSED' if code == 0 else 'FAILED'}")
        lines.append(out)
        lines.append("-" * 50)
        if code != 0:
            all_passed = False
    else:
        lines.append("[PIP-AUDIT] Não instalado no ambiente (ignorado)")
        lines.append("-" * 50)

    report_content = "\n".join(lines) + "\n"
    out_file = Path("reports/check.txt")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(report_content, encoding="utf-8")

    print(f"\nResultado gravado em {out_file}:")
    print(report_content)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
