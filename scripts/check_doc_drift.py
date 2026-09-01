#!/usr/bin/env python3
# check_doc_drift.py - 检查 CLI flag / yaml 字段是否已同步到文档。
# 解析 src/cli.py 的 add_argument("--xxx") 与 src/config.py 读到的 yaml 字段
# (PRECISION_CRITICAL_PARAMS + DETERMINISTIC_ENV + 顶层 model_yaml.get 键)，
# 核对是否出现在 README.md 或 models/_template.yaml。exit 0=同步, 1=漂移。
#
# 用法: python3 scripts/check_doc_drift.py   (或 bash run.sh scripts/check_doc_drift.py)
# 加入 CI / pre-push，保证"功能更新及时同步文档"。
import re, sys, os
KIT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def read(p):
    try: return open(p, encoding="utf-8").read()
    except FileNotFoundError: return ""
cli = read(os.path.join(KIT, "src/cli.py"))
cfg = read(os.path.join(KIT, "src/config.py"))
docs = read(os.path.join(KIT, "README.md")) + "\n" + read(os.path.join(KIT, "models/_template.yaml"))

# 1) CLI flags from cli.py add_argument("--xxx")
flags = sorted(set(re.findall(r'add_argument\(\s*"(-{1,2}[\w-]+)"', cli)))
miss_flag = [f for f in flags if f not in docs]

# 2) yaml fields read by config.py
fields = set()
m = re.search(r'PRECISION_CRITICAL_PARAMS[^{]*\{(.+?)\n\}', cfg, re.S)
if m: fields |= set(re.findall(r'"([a-z_]+)":\s*\(', m.group(1)))
m = re.search(r'DETERMINISTIC_ENV[^{]*\{(.+?)\n\}', cfg, re.S)
if m: fields |= set(re.findall(r'"([A-Z_]+)":', m.group(1)))
fields |= set(re.findall(r'model_yaml\.get\(\s*"([a-z_]+)"', cfg))
miss_field = [f for f in sorted(fields) if f not in docs]

ok = True
if miss_flag:
    ok = False; print("[drift] cli.py 的 flag 未出现在 README/_template.yaml：")
    for f in miss_flag: print("  -", f)
if miss_field:
    ok = False; print("[drift] config.py 的 yaml 字段未出现在 README/_template.yaml：")
    for f in miss_field: print("  -", f)
if ok: print("[drift] OK: cli flag 与 yaml 字段均已出现在文档。")
sys.exit(0 if ok else 1)
