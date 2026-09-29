# jd_agent 包结构

包内按职责分层，依赖方向固定为 `cli -> agents -> memory / knowledge / domain / services -> core`。移动模块时优先遵守这个方向，避免重新把解析、联网、编排和展示混在根目录。

```text
jd_agent/
├── __init__.py
├── cli.py                 # 命令行入口：参数、路径、模式分派、终端输出
├── core/                  # 无业务编排的共享基础
│   ├── schema.py          # 数据模型、状态与常量
│   ├── settings.py        # .env 与 LLM 配置
│   └── llm.py             # OpenAI 兼容客户端
├── memory/                # L0 会话 / L1 短时 / L2 长时记忆
│   ├── interfaces.py      # 稳定接口；L1 的依赖边界在这里
│   ├── short_term.py      # MVP 内存字典；换 Redis 只改这个文件
│   ├── long_term.py       # JSONL 默认实现 + 可选 MongoDB
│   ├── session.py         # L0 会话与 LangGraph checkpointer 适配
│   └── manager.py         # 三层记忆编排与固化
├── knowledge/             # L1 静态 / L2 半静态 / L3 动态知识
│   ├── interfaces.py      # EmbeddingProvider / VectorIndex 抽象
│   ├── embedding.py       # 本地哈希嵌入 + 通义 text-embedding-v4
│   ├── local_index.py     # 默认 JSONL 向量索引
│   ├── qdrant_index.py    # 可选 Qdrant 索引
│   └── manager.py         # 分层切片、增量索引与检索
├── domain/                # 纯业务规则与结构化解析
│   ├── jd.py              # JD 文本解析
│   ├── jd_html.py         # HTML -> Markdown / 正文清理
│   ├── project.py         # 项目描述解析
│   ├── resume_facts.py    # 简历事实、否定与证据等级
│   ├── evidence.py        # JD 要求与项目证据匹配
│   └── lexicon.py         # 能力词典
├── services/              # 外部 IO 与可选增强
│   ├── jd_source.py       # 文本/文件/图片/网址来源与抓取
│   └── vision.py          # 图片引用解析与 Qwen-VL 取证
├── agents/                # Agent 运行链与产物生成
│   ├── agent.py           # 项目是否值得写进简历的主 Agent
│   ├── jd_graph.py        # JD Agent
│   ├── resume_graph.py    # Resume Agent
│   ├── polish_graph.py    # Polish Agent
│   ├── generate.py        # 主 Agent 的可选 LLM 增强
│   ├── polish_brief.py    # Polish 的规则简报
│   ├── polish_llm.py      # Polish 的模型调用与解析
│   ├── render.py          # 主 Agent 的 md/json/html/console 输出
│   └── graph_view.py      # 图可视化辅助
└── tools/                 # 可独立调用的确定性工具
```

## 导入约定

- 命令行和前端只从具体分层导入，不通过根包集中导入业务对象。
- `core` 不依赖其它业务层。
- `domain` 可以依赖 `core`，不做 HTTP 请求或读取模型配置。
- `services` 可以依赖 `core`、`domain`。
- `memory` / `knowledge` 只依赖 `core`，具体后端通过接口与 factory 注入。
- `agents` 可以依赖 `memory`、`knowledge`、`domain`、`services`、`core` 和 `tools`。
- `tools` 只依赖 `core`、`domain` 和必要的 Agent 简报能力，不依赖 `cli`。

示例：

```python
from jd_agent.agents.agent import run_agent
from jd_agent.agents.jd_graph import run_jd_agent
from jd_agent.domain.jd import parse_jd_text
from jd_agent.core.schema import JobPosting
```
