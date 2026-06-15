"""Unit tests for the 6 fixes."""
from pathlib import Path
from autoscientist.tools.write_file import write_file, SandboxEscape
import tempfile, shutil

# Test 1: basic write
tmp = Path(tempfile.mkdtemp())
result = write_file(path='src/config.py', content='X = 42\n', project_id='test_wf', projects_root=tmp)
assert result['written'] is True
assert result['path'] == 'src/config.py'
written = (tmp / 'test_wf' / 'sandbox' / 'src' / 'config.py').read_text()
assert written == 'X = 42\n', f'got {written!r}'
print('PASS  write_file basic write')

# Test 2: sandbox escape blocked
try:
    write_file(path='../../../etc/passwd', content='pwned', project_id='test_wf', projects_root=tmp)
    print('FAIL  sandbox escape not caught')
except SandboxEscape:
    print('PASS  write_file blocks sandbox escape')

# Test 3: absolute path blocked
try:
    write_file(path='/tmp/evil.py', content='pwned', project_id='test_wf', projects_root=tmp)
    print('FAIL  absolute path not caught')
except SandboxEscape:
    print('PASS  write_file blocks absolute path')

# Test 4: tool registry has write_file
from autoscientist.tools.registry import ALL_TOOL_NAMES
assert 'write_file' in ALL_TOOL_NAMES, f'write_file not in {ALL_TOOL_NAMES}'
print('PASS  write_file registered in tool registry')

# Test 5: reasoning loop detection
from autoscientist.clients.ollama import _detect_reasoning_loop
block = 'Let me think about this. ' * 50  # 1250 chars
loop = block * 4  # same block repeated 4 times
assert _detect_reasoning_loop(loop) is True, 'should detect loop'
print('PASS  _detect_reasoning_loop detects repeated blocks')

no_loop = 'A' * 500 + 'B' * 500 + 'C' * 500 + 'D' * 500
assert _detect_reasoning_loop(no_loop) is False, 'should not false-positive'
print('PASS  _detect_reasoning_loop no false positive on varied text')

# Test 6: per-agent max_tool_rounds
from autoscientist.runtime.runner import _agent_max_tool_rounds
from autoscientist.runtime.config import load_config
cfg = load_config(reload=True)
assert _agent_max_tool_rounds(cfg, 'code_gen', 40) == 12, f'expected 12'
assert _agent_max_tool_rounds(cfg, 'test_gen', 40) == 8, f'expected 8'
assert _agent_max_tool_rounds(cfg, 'lit_review', 40) == 40, f'expected default 40'
print('PASS  per-agent max_tool_rounds reads from config')

# Test 7: project budget functions
from autoscientist.runtime.budget import project_spent, assert_project_budget, BudgetExceeded
from autoscientist.state.db import open_db
db_path = Path(tempfile.mktemp(suffix='.db'))
conn = open_db(db_path)
assert project_spent(conn, 'test_project') == 0.0
print('PASS  project_spent returns 0 for empty DB')

assert_project_budget(conn, 'test_project', 25.0, 1.0)
print('PASS  assert_project_budget passes under cap')

assert_project_budget(conn, 'test_project', 0.0, 999.0)
print('PASS  assert_project_budget skips when cap=0')

conn.close()
db_path.unlink()
shutil.rmtree(tmp)
print()
print('*** All unit checks passed. ***')
