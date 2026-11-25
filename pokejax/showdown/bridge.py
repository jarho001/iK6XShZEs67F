"""The line protocol with reference/showdown-player.cjs: one Node process per run.

Python sends one {"type": "start"} message, then one {"type": "actions"} message for each decisions
message Node sends, until Node sends {"type": "done"}. The messages are described at the top of the
script. Node's standard error goes to this process's standard error.
"""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "reference" / "showdown-player.cjs"


class BridgeError(RuntimeError):
	"""The Node side failed, or sent a message this side does not understand."""


class Bridge:
	"""One Node process playing the battles of one run (a list of {index, teams, learner})."""

	def __init__(
		self,
		pool_json: Path,
		battles: list[dict],
		*,
		opponent: str,
		seed: int,
		max_steps: int,
		concurrency: int,
		record: bool = False,
		search: bool = False,
		audit: bool = False,
	):
		self.process = subprocess.Popen(
			["node", str(SCRIPT)],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			encoding="utf-8",
			cwd=ROOT,
		)
		self._write(
			{
				"type": "start",
				"pool": str(Path(pool_json).resolve()),
				"battles": battles,
				"opponent": opponent,
				"seed": seed,
				"maxSteps": max_steps,
				"concurrency": concurrency,
				"record": record,
				"search": search,
				"audit": audit,
			}
		)

	def _write(self, message: dict) -> None:
		self.process.stdin.write(json.dumps(message) + "\n")
		self.process.stdin.flush()

	def receive(self) -> dict:
		"""The next message from Node: decisions, or done with the last results."""
		line = self.process.stdout.readline()
		if not line:
			raise BridgeError(f"node exited with status {self.process.wait()} before finishing")
		message = json.loads(line)
		if message["type"] == "fatal":
			raise BridgeError(message["error"])
		return message

	def answer(self, items: list[dict], actions) -> None:
		"""Send one action per decision item, in the order of the items."""
		self._write(
			{
				"type": "actions",
				"items": [
					{
						"slot": item["slot"],
						"action": int(action),
						**({"side": item["side"]} if "side" in item else {}),
					}
					for item, action in zip(items, actions, strict=True)
				],
			}
		)

	def close(self) -> int:
		self.process.stdin.close()
		return self.process.wait()
