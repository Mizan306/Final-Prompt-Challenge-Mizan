# AI-Powered Task Manager (Codespaces)

To-Do + AI assistant. Runs great in **GitHub Codespaces**.

- **Backend:** Flask (in-memory store) + `/api/ai` assistant
- **Frontend:** Single-page HTML/CSS/JS with chat panel
- **AI mode:** Uses OpenAI if `OPENAI_API_KEY` is set; otherwise falls back to a rule-based parser.

## Quickstart (Codespaces)

```bash
pip install -r requirements.txt
python app.py
```
Open the forwarded port (usually **5000**). The UI shows a **mode** badge.

## Enable the LLM Assistant

1. In Codespaces, set your secret **OPENAI_API_KEY** (preferably via Codespaces Secrets).  
2. (Optional) Choose a model:
   ```bash
   export OPENAI_MODEL=gpt-4o-mini
   ```
3. Restart the app:
   ```bash
   python app.py
   ```
The `/api/mode` endpoint (and the badge in the header) will show `mode: llm` when active.

## REST API

- `POST /api/tasks` → Create a task
  ```json
  {"description": "buy milk"}
  ```
- `GET /api/tasks` → List all tasks
- `PATCH /api/tasks/<id>/complete` → Mark complete
- `DELETE /api/tasks/<id>` → Delete

## Assistant API

- `POST /api/ai`
  ```json
  { "message": "done with task number 2", "execute": true }
  ```
  **Response**
  ```json
  {
    "mode": "llm",
    "decision": {"function":"completeTask","parameters":{"task_id":2}},
    "executed": true,
    "result": { "id": 2, "description": "...", "completed": true, "created_at": 1711111111.0 }
  }
  ```

## Notes
- Data is **in-memory**; restarts reset tasks.
- You can later swap in SQLite + SQLAlchemy for persistence.
- The rule-based fallback understands ordinals (“third”), numerals (`#3`, `id 2`), and quotes for descriptions.
