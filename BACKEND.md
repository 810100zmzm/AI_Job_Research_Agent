# FastAPI 后端

启动：

```powershell
uvicorn jd_agent.api.app:app --host 127.0.0.1 --port 8000 --reload
```

接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| `POST` | `/api/upload` | 上传文件，转换成 Markdown 并返回 `file_id` |
| `GET` | `/api/input-files?kind=jd` | 列出 `input/jd` 或 `input/profile` 中可选文件 |
| `POST` | `/api/input-files/select` | 将选中的 input 文件转换成 Markdown 并返回 `file_id` |
| `POST` | `/api/analyze` | 创建后台分析任务，立即返回 `task_id` |
| `GET` | `/api/chat?task_id=...` | 以 SSE 推送 Agent Trace |
| `GET` | `/api/report/{task_id}?format=md` | 读取 Markdown / JSON / HTML 报告 |
| `POST` | `/api/records` | 新增 L2 求职记录 |
| `GET` | `/api/records?status=...` | 查询 L2 求职记录 |
| `DELETE` | `/api/records/{record_id}` | 删除 L2 求职记录 |
| `GET` | `/api/resume/options` | 获取 L1/L2/L3 表达强度与三套简历风格 |
| `POST` | `/api/resume/polish` | 规则优先润色，生成润色版简历与原文对照 |
| `GET` | `/api/resume/{resume_id}?format=md` | 读取生成的 Markdown / HTML 简历 |

`/api/analyze` 的 `jd_input` 和 `resume_input` 既可以直接传文本，也可以传
`/api/input-files` 只暴露 `input/jd` 与 `input/profile` 两个目录；`kind=jd` 对应
JD，`kind=resume` 对应简历。选择接口返回的 `file_id` 可直接传给 `/api/analyze`。

`/api/upload` 返回的 `file_id`。报告写入 `output/<task_id>.md|json|html`，
上传临时文件写入 `data/tmp/uploads/`，L2 记录复用 `data/memory/long_term.jsonl`
或现有 MongoDB 后端。

`/api/resume/polish` 可传已完成分析的 `task_id`，也可直接传 `resume_input`（文本或上传 `file_id`）。
响应中的 `items[]` 每条都带 `source: llm-polish`，并通过 `engine` 区分 `dictionary` /
`llm` / `original`；模型不可用时逐条回落原文，Markdown 与 HTML 仍然生成。
