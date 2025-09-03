#!/usr/bin/env python3
"""
AI-Powered Task Manager - Flask Backend (LLM + Rule-based, Chatty UI)
--------------------------------------------------------------------
- To-Do List API (in-memory) with priorities
- Assistant endpoint /api/ai: conversational replies + optional action execution
  Uses OpenAI if OPENAI_API_KEY is set; else falls back to a rules engine.

Env:
  OPENAI_API_KEY  (required for LLM mode)
  OPENAI_MODEL    (optional; defaults to gpt-3.5-turbo)

Run:
  pip install -r requirements.txt
  python app.py
"""
from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Dict, List
import os, time, re, json

# =============================================================================
# Data model & storage
# =============================================================================
@dataclass
class Task:
    id: int
    description: str
    completed: bool = False
    priority: str = "medium"   # only "low", "medium", "high"
    created_at: float = time.time()

class TaskStore:
    """Thread-safe in-memory task storage."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1
        self._completed: List[Task] = []
        self._deleted: List[Task] = []

    def add(self, description: str, priority: str = "medium") -> Task:
        if priority not in ["low", "medium", "high"]:
            priority = "medium"
        with self._lock:
            task = Task(
                id=self._next_id,
                description=description,
                completed=False,
                priority=priority,
                created_at=time.time()
            )
            self._tasks[self._next_id] = task
            self._next_id += 1
            return task

    def all(self) -> List[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.created_at)

    def complete(self, task_id: int) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                task.completed = True
                self._completed.append(task)
            return task

    def delete(self, task_id: int) -> bool:
        with self._lock:
            task = self._tasks.pop(task_id, None)
            if task:
                self._deleted.append(task)
                return True
            return False

    def get(self, task_id: int) -> Task | None:
        with self._lock:
            return self._tasks.get(task_id)

    # ---- Name-based helpers ----
    def find_by_name(self, name: str) -> Task | None:
        with self._lock:
            for t in self._tasks.values():
                if t.description.lower() == name.lower():
                    return t
        return None

    def complete_by_name(self, name: str) -> Task | None:
        task = self.find_by_name(name)
        if task:
            task.completed = True
            self._completed.append(task)
        return task

    def delete_by_name(self, name: str) -> bool:
        task = self.find_by_name(name)
        if task:
            with self._lock:
                self._deleted.append(task)
                return self._tasks.pop(task.id, None) is not None
        return False

    def history(self) -> dict:
        with self._lock:
            return {
                "completed": [asdict(t) for t in self._completed],
                "deleted": [asdict(t) for t in self._deleted]
            }

store = TaskStore()

# =============================================================================
# Rule-based assistant (fallback)
# =============================================================================
class RuleBrain:
    """Maps NL → {function, parameters} or returns None for pure chat."""
    def parse(self, text: str) -> Dict | None:
        t = text.strip()
        if not t:
            raise ValueError("Empty message")
        low = t.lower()
        words = t.split(maxsplit=1)

        # --- Add ---
        if low.startswith("add "):
            if len(words) < 2:
                raise ValueError("Missing description after 'add'")
            desc = words[1].strip()
            prio = "medium"
            if desc.endswith(" high"):
                desc = desc.replace(" high", "").strip()
                prio = "high"
            elif desc.endswith(" low"):
                desc = desc.replace(" low", "").strip()
                prio = "low"
            elif desc.endswith(" medium"):
                desc = desc.replace(" medium", "").strip()
                prio = "medium"
            return {"function": "addTask", "parameters": {"description": desc, "priority": prio}}

        # --- Complete ---
        if low.startswith("complete "):
            if len(words) < 2:
                raise ValueError("Missing task name after 'complete'")
            return {"function": "completeTask", "parameters": {"task_name": words[1].strip()}}

        # --- Delete ---
        if low.startswith("delete "):
            if len(words) < 2:
                raise ValueError("Missing task name after 'delete'")
            return {"function": "deleteTask", "parameters": {"task_name": words[1].strip()}}

        # --- View ---
        if low in ["show", "list", "view", "tasks", "todo", "what do i have"]:
            return {"function": "viewTasks", "parameters": {}}

        return None

# =============================================================================
# LLM assistant (OpenAI) - JSON-only with fallback
# =============================================================================
class LLMBrain:
    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.system = (
            "You output ONLY one JSON object with a function call from:\n"
            "addTask(description: string, priority: string = 'medium')\n"
            "viewTasks()\n"
            "completeTask(task_name: string)\n"
            "deleteTask(task_name: string)\n"
            "Valid priority values: 'low', 'medium', 'high'.\n"
            'Format: {\"function\":\"...\",\"parameters\":{...}}'
        )

    def parse(self, user_text: str) -> dict | None:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": user_text}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            return json.loads(content)
        except Exception:
            return None

# =============================================================================
# Flask app
# =============================================================================
app = Flask(__name__, static_folder='.', static_url_path='')

rule_brain = RuleBrain()
llm_brain = None
if os.environ.get("OPENAI_API_KEY"):
    try:
        llm_brain = LLMBrain()
    except Exception:
        llm_brain = None

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# ============================ Tasks API ============================
@app.post('/api/tasks')
def create_task():
    data = request.get_json(silent=True) or {}
    description = (data.get('description') or '').strip()
    priority = (data.get('priority') or 'medium').lower()
    if not description:
        return jsonify({'error': 'description is required'}), 400
    if priority not in ["low", "medium", "high"]:
        priority = "medium"
    task = store.add(description, priority)
    return jsonify(asdict(task)), 201

@app.get('/api/tasks')
def list_tasks():
    tasks = [asdict(t) for t in store.all()]
    return jsonify(tasks), 200

@app.get('/api/history')
def history():
    return jsonify(store.history()), 200

@app.patch('/api/tasks/<int:task_id>/complete')
def complete_task(task_id: int):
    task = store.complete(task_id)
    if not task:
        return jsonify({'error': f'task {task_id} not found'}), 404
    return jsonify(asdict(task)), 200

@app.delete('/api/tasks/<int:task_id>')
def delete_task(task_id: int):
    ok = store.delete(task_id)
    if not ok:
        return jsonify({'error': f'task {task_id} not found'}), 404
    return ('', 204)

# ============================ Assistant API ============================
def _friendly_reply(decision: dict | None, result, message: str) -> str:
    if decision is None:
        return "I'm here to help with your tasks. Try: add buy milk or list"
    fn = decision.get("function")
    p = decision.get("parameters", {})
    if fn == "addTask":
        return f'Added: {p.get("description")} [{p.get("priority","medium")}] 👍'
    if fn == "viewTasks":
        return f"Here are your tasks ({len(result or [])})."
    if fn == "completeTask":
        return f"Completed task {p.get('task_name')} ✅"
    if fn == "deleteTask":
        return f"Deleted task {p.get('task_name')} 🗑"
    return "Done."

@app.post('/api/ai')
def ai_route():
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    execute = bool(data.get('execute', True))
    if not message:
        return jsonify({'error': 'message is required'}), 400

    decision = None
    if llm_brain:
        decision = llm_brain.parse(message)
    if decision is None:
        decision = rule_brain.parse(message)

    result = None
    status = 200
    if execute and decision:
        fn = decision.get('function')
        params = decision.get('parameters', {})
        if fn == 'addTask':
            desc = params.get('description')
            prio = (params.get('priority') or "medium").lower()
            if prio not in ["low", "medium", "high"]:
                prio = "medium"
            result = asdict(store.add(desc, prio))
            status = 201
        elif fn == 'viewTasks':
            result = [asdict(t) for t in store.all()]
        elif fn == 'completeTask':
            task = store.complete_by_name(params.get('task_name'))
            if not task:
                return jsonify({'error': f'task {params.get("task_name")!r} not found'}), 404
            result = asdict(task)
        elif fn == 'deleteTask':
            ok = store.delete_by_name(params.get('task_name'))
            if not ok:
                return jsonify({'error': f'task {params.get("task_name")!r} not found'}), 404
            result = {'deleted': params.get('task_name')}

    reply = _friendly_reply(decision, result, message)
    return jsonify({'decision': decision, 'executed': bool(execute and decision),
                    'result': result, 'reply': reply}), status

# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
