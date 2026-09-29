"""End-to-end smoke tests for the FastAPI backend skeleton."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from jd_agent.api.app import create_app

JD_TEXT = """# 后端开发工程师

公司：示例科技

## 岗位职责

- 负责 Python 后端服务与接口开发

## 任职要求

- 熟悉 Python 与 FastAPI
- 熟悉 Redis 缓存
"""

RESUME_TEXT = """# 项目经历

## AI 周报助手

- 使用 Python 和 FastAPI 搭建摘要接口，项目已上线，响应时间从 800ms 降到 120ms
- 使用 Redis 缓存高频结果，后续请求延迟降低 60%
"""

THIN_RESUME_TEXT = """# 项目经历

## 数据平台

- 了解 Python 后端开发
- 参与过接口相关工作
"""


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        input_dir = root / "input"
        (input_dir / "jd" / "nested").mkdir(parents=True)
        (input_dir / "profile").mkdir(parents=True)
        (input_dir / "jd" / "jd.md").write_text(JD_TEXT, encoding="utf-8")
        (input_dir / "jd" / "nested" / "jd2.md").write_text(JD_TEXT, encoding="utf-8")
        (input_dir / "jd" / "unsupported.exe").write_bytes(b"no")
        (input_dir / "profile" / "resume.md").write_text(RESUME_TEXT, encoding="utf-8")
        frontend_dir = root / "frontend-dist"
        frontend_dir.mkdir()
        (frontend_dir / "index.html").write_text(
            "<!doctype html><div id=\"app\">frontend test shell</div>",
            encoding="utf-8",
        )
        app = create_app(
            data_dir=root / "data",
            output_dir=root / "output",
            input_dir=input_dir,
            frontend_dir=frontend_dir,
            image_transcriber=lambda path: f"# 截图转录\n\n- {path.name}",
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.tmp.cleanup()

    def test_input_files_can_be_listed_selected_and_analyzed(self) -> None:
        listed_jd = self.client.get("/api/input-files", params={"kind": "jd"})
        self.assertEqual(listed_jd.status_code, 200, listed_jd.text)
        jd_items = listed_jd.json()
        self.assertEqual([item["path"] for item in jd_items], ["jd.md", "nested/jd2.md"])
        self.assertEqual(jd_items[0]["label"], "jd/jd.md")
        self.assertNotIn("unsupported.exe", [item["path"] for item in jd_items])

        selected_jd = self.client.post(
            "/api/input-files/select",
            json={"kind": "jd", "path": "jd.md"},
        )
        self.assertEqual(selected_jd.status_code, 200, selected_jd.text)
        self.assertEqual(selected_jd.json()["filename"], "jd.md")

        selected_resume = self.client.post(
            "/api/input-files/select",
            json={"kind": "resume", "path": "resume.md"},
        )
        self.assertEqual(selected_resume.status_code, 200, selected_resume.text)

        analyze = self.client.post(
            "/api/analyze",
            json={
                "jd_input": selected_jd.json()["file_id"],
                "resume_input": selected_resume.json()["file_id"],
                "session_id": "input-file-test",
                "user_id": "tester",
            },
        )
        self.assertEqual(analyze.status_code, 200, analyze.text)
        stream = self.client.get("/api/chat", params={"task_id": analyze.json()["task_id"]})
        self.assertEqual(stream.status_code, 200, stream.text)
        self.assertIn("event: done", stream.text)

        traversal = self.client.post(
            "/api/input-files/select",
            json={"kind": "jd", "path": "../profile/resume.md"},
        )
        self.assertEqual(traversal.status_code, 400, traversal.text)

    def test_frontend_static_mount_and_spa_fallback(self) -> None:
        root = self.client.get("/")
        self.assertEqual(root.status_code, 200, root.text)
        self.assertIn("frontend test shell", root.text)

        history = self.client.get("/history")
        self.assertEqual(history.status_code, 200, history.text)
        self.assertIn("frontend test shell", history.text)

        missing_api = self.client.get("/api/not-found")
        self.assertEqual(missing_api.status_code, 404)
        self.assertEqual(missing_api.json(), {"detail": "Not Found"})

    def test_image_upload_uses_vision_transcriber(self) -> None:
        uploaded = self.client.post(
            "/api/upload",
            files={"file": ("jd.png", b"fake-image", "image/png")},
        )
        self.assertEqual(uploaded.status_code, 200, uploaded.text)
        payload = uploaded.json()
        self.assertEqual(payload["file_type"], "image")
        self.assertIn("jd.png", payload["md_content"])

    def test_upload_analyze_stream_and_reports(self) -> None:
        jd = self.client.post(
            "/api/upload",
            files={"file": ("jd.md", JD_TEXT.encode("utf-8"), "text/markdown")},
        )
        self.assertEqual(jd.status_code, 200, jd.text)
        jd_payload = jd.json()
        self.assertTrue(jd_payload["file_id"])
        self.assertEqual(jd_payload["file_type"], "text")

        resume = self.client.post(
            "/api/upload",
            files={"file": ("resume.md", RESUME_TEXT.encode("utf-8"), "text/markdown")},
        )
        self.assertEqual(resume.status_code, 200, resume.text)

        analyze = self.client.post(
            "/api/analyze",
            json={
                "jd_input": jd_payload["file_id"],
                "resume_input": resume.json()["file_id"],
                "session_id": "api-test",
                "user_id": "tester",
            },
        )
        self.assertEqual(analyze.status_code, 200, analyze.text)
        task_id = analyze.json()["task_id"]
        self.assertTrue(task_id)

        stream = self.client.get("/api/chat", params={"task_id": task_id})
        self.assertEqual(stream.status_code, 200, stream.text)
        self.assertIn('"action": "ReadJD"', stream.text)
        self.assertIn("event: done", stream.text)

        markdown = self.client.get(f"/api/report/{task_id}", params={"format": "md"})
        self.assertEqual(markdown.status_code, 200, markdown.text)
        self.assertIn("运行 Trace", markdown.text)

        payload = self.client.get(f"/api/report/{task_id}", params={"format": "json"})
        self.assertEqual(payload.status_code, 200, payload.text)
        self.assertEqual(payload.json()["trace"][0]["action"], "ReadJD")

        html = self.client.get(f"/api/report/{task_id}", params={"format": "html"})
        self.assertEqual(html.status_code, 200, html.text)
        self.assertIn("<!DOCTYPE html>", html.text)

    def test_ask_can_be_answered_and_then_generate_report(self) -> None:
        analyze = self.client.post(
            "/api/analyze",
            json={
                "jd_input": JD_TEXT,
                "resume_input": THIN_RESUME_TEXT,
                "session_id": "ask-test",
                "user_id": "tester",
            },
        )
        self.assertEqual(analyze.status_code, 200, analyze.text)
        task_id = analyze.json()["task_id"]

        first_stream = self.client.get("/api/chat", params={"task_id": task_id})
        self.assertEqual(first_stream.status_code, 200, first_stream.text)
        self.assertIn("event: ask", first_stream.text)
        self.assertIn('"question"', first_stream.text)
        self.assertNotIn("event: done", first_stream.text)
        self.assertEqual(
            self.client.get(f"/api/report/{task_id}").status_code,
            404,
        )

        answer = self.client.post(
            f"/api/task/{task_id}/answer",
            json={
                "answer": (
                    "数据平台由我独立开发：使用 Python 和 FastAPI 搭建了后端接口，"
                    "并用 Redis 缓存高频查询；接口响应时间从 800ms 降到 120ms。"
                )
            },
        )
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(answer.json()["status"], "running")

        second_stream = self.client.get("/api/chat", params={"task_id": task_id})
        self.assertEqual(second_stream.status_code, 200, second_stream.text)
        self.assertIn('"action": "ApplyUserAnswer"', second_stream.text)
        self.assertIn("event: done", second_stream.text)

        payload = self.client.get(f"/api/report/{task_id}", params={"format": "json"})
        self.assertEqual(payload.status_code, 200, payload.text)
        report = payload.json()
        self.assertEqual(report["decision"], "Stop")
        actions = [item["action"] for item in report["trace"]]
        self.assertIn("ApplyUserAnswer", actions)

        duplicate = self.client.post(
            f"/api/task/{task_id}/answer",
            json={"answer": "重复回答"},
        )
        self.assertEqual(duplicate.status_code, 409, duplicate.text)

    def test_l2_records_create_list_filter_and_delete(self) -> None:
        created = self.client.post(
            "/api/records",
            json={
                "company": "示例科技",
                "position": "后端开发工程师",
                "location": "上海",
                "status": "interviewing",
                "match_score": 82.5,
                "note": "二面准备系统设计",
            },
        )
        self.assertEqual(created.status_code, 200, created.text)
        record_id = created.json()["record_id"]

        listed = self.client.get("/api/records", params={"status": "interviewing"})
        self.assertEqual(listed.status_code, 200, listed.text)
        items = listed.json()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["record_id"], record_id)
        self.assertEqual(items[0]["match_score"], 82.5)

        empty = self.client.get("/api/records", params={"status": "rejected"})
        self.assertEqual(empty.status_code, 200, empty.text)
        self.assertEqual(empty.json(), [])

        updated = self.client.put(
            f"/api/records/{record_id}",
            json={
                "company": "示例科技",
                "position": "高级后端开发工程师",
                "location": "杭州",
                "status": "offer",
                "match_score": 91.0,
                "note": "已谈薪",
            },
        )
        self.assertEqual(updated.status_code, 200, updated.text)
        self.assertEqual(updated.json()["record_id"], record_id)
        self.assertEqual(updated.json()["position"], "高级后端开发工程师")
        self.assertEqual(updated.json()["match_score"], 91.0)

        old_filter = self.client.get("/api/records", params={"status": "interviewing"})
        self.assertEqual(old_filter.status_code, 200, old_filter.text)
        self.assertEqual(old_filter.json(), [])

        new_filter = self.client.get("/api/records", params={"status": "offer"})
        self.assertEqual(new_filter.status_code, 200, new_filter.text)
        self.assertEqual(new_filter.json()[0]["note"], "已谈薪")

        deleted = self.client.delete(f"/api/records/{record_id}")
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json(), {"success": True})


if __name__ == "__main__":
    unittest.main()
