#!/usr/bin/env python3
"""
AI-Powered Task Manager - Flask Backend (LLM + Rule-based, Chatty UI)
--------------------------------------------------------------------
- To-Do List API (in-memory)
- Assistant endpoint /api/ai: conversational replies + optional action execution
  Uses OpenAI if OPENAI_API_KEY is set; else falls back to a rules engine.
- Mode endpoint /api/mode: which brain is active (no-store).

Env:
  OPENAI_API_KEY  (required for LLM mode)
  OPENAI_MODEL    (optional; defaults to gpt-3.5-turbo)

Run:
  pip install -r requirements.txt
  python app.py
"""
from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory, make_response
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Dict, List
import os, time, re, json

# ----------------------------------------------------------------------------
# Data model & storage
# ----------------------------------------------------------------------------
@dataclass
class Task:
    id: int
    description: str
    completed: bool = False
    created_at: float = time.time()

class TaskStore:
    """Thread-safe in-memory task storage."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._tasks: Dict[int, Task] = {}
        self._next_id = 1

    def add(self, description: str) -> Task:
        with self._lock:
            task = Task(id=self._next_id, description=description, completed=False, created_at=time.time())
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
            return task

    def delete(self, task_id: int) -> bool:
        with self._lock:
            return self._tasks.pop(task_id, None) is not None

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
        return task

    def delete_by_name(self, name: str) -> bool:
        task = self.find_by_name(name)
        if task:
            with self._lock:
                return self._tasks.pop(task.id, None) is not None
        return False

store = TaskStore()

# ----------------------------------------------------------------------------
# Rule-based assistant (fallback)
# ----------------------------------------------------------------------------
class RuleBrain:
    """Maps NL → {function, parameters} or returns None for pure chat."""
    def parse(self, text: str) -> Dict | None:
        t = text.strip()
        if not t:
            raise ValueError("Empty message")
        low = t.lower()

        # Conversational (no action)
        if re.search(r'\b(hi|hello|hey|good (morning|afternoon|evening))\b', low):
            return None
        if "what can you do" in low or "help" in low:
            return None

        words = t.split(maxsplit=1)

        # --- Add ---
        if low.startswith("add "):
            if len(words) < 2:
                raise ValueError("Missing description after 'add'")
            return {"function": "addTask", "parameters": {"description": words[1].strip()}}

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

        # Fallback: no action
        return None

# ----------------------------------------------------------------------------
# LLM assistant (OpenAI) - JSON-only with fallback
# ----------------------------------------------------------------------------
class LLMBrain:
    def __init__(self) -> None:
        from openai import OpenAI
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo")
        self.system = (
            "You output ONLY one JSON object with a function call from:\n"
            "addTask(description: string)\n"
            "viewTasks()\n"
            "completeTask(task_name: string)\n"
            "deleteTask(task_name: string)\n"
            'Format: {"function":"...","parameters":{...}}'
        )

    def parse(self, user_text: str) -> dict | None:
        if re.search(r'\b(hi|hello|hey|good (morning|afternoon|evening))\b', user_text.lower()) or \
           ("what can you do" in user_text.lower() or "help" in user_text.lower()):
            return None
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": user_text}],
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = (resp.choices[0].message.content or "").strip()
            data = json.loads(content)
            if not isinstance(data, dict) or "function" not in data or "parameters" not in data:
                raise ValueError("Invalid JSON shape")
            return data
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.system},
                          {"role": "user", "content": user_text}],
                temperature=0,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = text.strip("`")
                if "\n" in text:
                    text = text.split("\n", 1)[1].strip()
            start = text.find("{"); end = text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError(f"LLM did not return JSON: {text!r}")
            data = json.loads(text[start:end+1])
            if not isinstance(data, dict) or "function" not in data or "parameters" not in data:
                raise ValueError(f"Invalid JSON shape: {data!r}")
            return data

# ----------------------------------------------------------------------------
# Flask app
# ----------------------------------------------------------------------------
app = Flask(__name__, static_folder='.', static_url_path='')

def llm_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))

rule_brain = RuleBrain()
llm_brain = None
if llm_available():
    try:
        llm_brain = LLMBrain()
    except Exception as e:
        import traceback; traceback.print_exc()
        print("LLM init failed:", e)
        llm_brain = None

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.get('/api/mode')
def get_mode():
    mode = "llm" if (llm_brain is not None) else "rules"
    model = os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo") if mode == "llm" else None
    resp = make_response(jsonify({"mode": mode, "model": model}))
    resp.headers["Cache-Control"] = "no-store"
    return resp

# --------------------------- Tasks API --------------------------------------
@app.post('/api/tasks')
def create_task():
    data = request.get_json(silent=True) or {}
    description = (data.get('description') or '').strip()
    if not description:
        return jsonify({'error': 'description is required'}), 400
    task = store.add(description)
    return jsonify(asdict(task)), 201

@app.get('/api/tasks')
def list_tasks():
    tasks = [asdict(t) for t in store.all()]
    return jsonify(tasks), 200

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

# --------------------------- Assistant API ----------------------------------
def _friendly_reply(decision: dict | None, result, message: str) -> str:
    low = message.lower()
    if decision is None:
        if re.search(r'\b(hi|hello|hey|good (morning|afternoon|evening))\b', low):
            return "Good morning! How can I assist you today?"
        if "what can you do" in low or "help" in low:
            return ('AI: "I can help you manage your tasks. You can ask me to add, '
                    'complete, or delete tasks from your to-do list. You can also '
                    'ask me to show you all your tasks."')
        return "I'm here to help with your tasks. Try: add buy milk or list"

    fn = decision.get("function")
    p = decision.get("parameters", {})
    if fn == "addTask":
        return f'Added: {p.get("description","(no description)")} 👍'
    if fn == "viewTasks":
        count = len(result or [])
        return f"Here are your tasks ({count})."
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

    try:
        if llm_brain is not None:
            decision = llm_brain.parse(message)
            brain_used = "llm"
        else:
            decision = rule_brain.parse(message)
            brain_used = "rules"
    except Exception as e:
        return jsonify({'error': str(e), 'mode': brain_used}), 400

    result = None
    status = 200
    if execute and decision:
        fn = decision.get('function')
        params = decision.get('parameters', {})
        if fn == 'addTask':
            desc = (params.get('description') or '').strip()
            if not desc:
                return jsonify({'error': 'description is required', 'mode': brain_used}), 400
            result = asdict(store.add(desc))
            status = 201
        elif fn == 'viewTasks':
            result = [asdict(t) for t in store.all()]
            status = 200
        elif fn == 'completeTask':
            tname = params.get('task_name')
            if not tname:
                return jsonify({'error': 'task_name required', 'mode': brain_used}), 400
            task = store.complete_by_name(tname)
            if not task:
                return jsonify({'error': f'task {tname!r} not found', 'mode': brain_used}), 404
            result = asdict(task)
            status = 200
        elif fn == 'deleteTask':
            tname = params.get('task_name')
            if not tname:
                return jsonify({'error': 'task_name required', 'mode': brain_used}), 400
            ok = store.delete_by_name(tname)
            if not ok:
                return jsonify({'error': f'task {tname!r} not found', 'mode': brain_used}), 404
            result = {'deleted': tname}
            status = 200

    reply = _friendly_reply(decision, result, message)
    return jsonify({'mode': brain_used, 'decision': decision, 'executed': bool(execute and decision),
                    'result': result, 'reply': reply}), status

# ----------------------------------------------------------------------------
# Entrypoint
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
