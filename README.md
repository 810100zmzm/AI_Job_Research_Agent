# AI 求职尽调 Agent

> **一个把「你做过什么」和「岗位要什么」对齐，回答「这个项目值不值得写进简历」的求职尽调 Agent。**
> 当前版本：**v4**

做过几个 AI 项目，却说不清哪个真正适合写进简历；更担心写进去之后，面试官顺着简历往下问，自己答不上来。

这个 Agent 只做两件事：

- **判断**：读「一份 JD + 一个项目描述」，抽出岗位真正需要的能力，到项目里逐条找能支撑这些能力的证据，给出**要不要写进简历**的判断（2~3 条理由 + 1~2 个风险），并留下可以逐条检查的运行 Trace。
- **润色**：基于**已核验的事实**，按 L1 保守 / L2 标准 / L3 亮眼三级把简历改得更专业、更匹配 JD。三级事实完全一样，只有说法不同：**不编数字、不编职责、不把「参与」写成「主导」。**

### 核心原则

> **LLM 负责想和行动，规则负责事实和证据。**

- LLM 决定下一步做什么、话怎么说；能不能当证据、够不够格下结论，全部由规则判定。
- 每条结论都能回到原文的 `文件::L行号`；资料不够就停下来问**一个**最关键的问题，不替用户猜。
- 默认**纯离线**：不加开关就不发任何请求，也不读 `.env`。

---

## 版本演进

### v1 · 立项：多份 JD 交叉比对

最初的形态是「多份 JD → 共同要求 → 对照简历找差距」，回答的是「这类岗位大概要什么」。
它能给出岗位画像，却没有回答用户真正卡住的那个问题。

### v2 · 主线成链：一个项目对一份 JD

主线收敛为一句话：**一个项目，对一份 JD，回答「写不写」。** 在规则判定之上，长出三条可审计的 Agent 链：

- **JD Agent**：把 JD（文本 / 文件 / 图片 / 网址）规整成统一模板的 Markdown；
- **Resume Agent**：把简历拆成一条条事实，先挡「写了但没做过」，再逐条判证据等级；
- **Polish Agent**：基于已核验事实 + JD 要求，产出建议写法 / 追问预演 / 润色。

同一版补齐了能力词典、工具区（`Tool` / `ToolResult` 统一契约）、简历排版，以及分层记忆与知识库。

### v3 · 全链路 Web 化

把主链完整搬到 Web 上：**Vue3 前端（输入 / 进度 / 报告 / 简历润色 / 历史）+ FastAPI 后端（SSE 流式 Trace）**。
上传 JD 与简历、实时看每一步、切换 md / json / html 三种报告、Ask 就地补答、一键生成润色版简历、顺手记投递。
后端只做包装、前端只做展示，核心判定逻辑一行没动，离线原则不变。

### v4 · 当前版本：Agent 循环与真实记忆

v4 把两条主线做扎实：**判定链从「固定流程图」升级为「LLM 驱动的有界 Agent 循环」**，
**记忆从「只写不读的展示层」升级为「真正参与规划的工作系统」**。

- 有界 **plan → act → observe** 循环：LLM 输出 `{action, args, reason}`，工具白名单执行，结果回灌，直到 Ask / Stop 或到达步数上限；现有多数节点降级为可被规划器调用的工具。
- 新增 **verify** 节点：LLM 组织的文字必须逐条对回证据，防幻觉。
- 三层记忆真正落地：**L0 可回放 / L1 接 checkpointer 与 Redis / L2 走向量检索**。
- 记忆读取注入 planner 上下文，运行后写回固化，并以 `recall` / `remember` 工具暴露给 Agent。

---

## v4 架构

### 1. 有界的 plan → act → observe 循环

判定不再是一张写死的流程图，而是一段有边界的自主行动：

1. **Plan**：LLM 读当前目标、上下文与可用工具，输出下一步 `{action, args, reason}`；
2. **Act**：只允许调用工具白名单里的工具，参数越界直接拦下；
3. **Observe**：把工具结果回灌进状态，作为下一步的输入。

循环在三种情况下结束：Agent 主动 `Ask`（缺只有用户才知道的事实）、主动 `Stop`（证据够了）、或到达最大步数上限。
每一步都带 `reason`，全程可审计。

**决策只有四种**：

| Decision | 什么时候用 |
|----------|-----------|
| `Continue` | 这一步的信息够了，按计划继续下一步 |
| `Adjust` | 搜索结果不理想，换一种口径再查一次（例如先严格后放宽） |
| `Ask` | 缺的是只有用户才知道的事实，停下来问**一个**问题 |
| `Stop` | 证据足够支撑结论，输出判断并写明 `StopReason` |

**每一步都留四样东西**：`Action`（做了什么）、`Observation`（实际看到了什么，带数字 / 文件名 / 行号）、`State Update`（状态被改成了什么）、`Decision`（下一步选哪个，为什么）。

### 2. verify 节点：防幻觉的最后一道闸

LLM 负责把证据组织成通顺的文字，但每一句都要回到规则给出的证据条目上复核：
对不上证据的表述被丢弃，缺证据的内容不进报告。**LLM 只负责「怎么说」，不负责「能不能写」。**

### 3. 双策略：规则打底，LLM 增强

| 策略 | 行为 | 适用 |
|------|------|------|
| `rule`（默认） | 纯规则判定，离线、可复现、测试稳定 | 日常使用、回归测试、无 Key 环境 |
| `llm` | LLM 驱动的 plan-act 循环 + verify | 需要更灵活的取证与表达时 |

两条策略共用同一套工具、同一套 Trace 格式与同一套判定阈值，切换不改结论口径。

### 4. 三层记忆：真正参与判定

| 层 | 存什么 | 实现 |
|----|--------|------|
| **L0 感知 / 输入缓冲** | 原始输入本身：原文、文件、图片、抓取的原始 HTML | 落盘可回放，随时回到第一手材料 |
| **L1 工作记忆** | 当前运行的目标、计划、工具结果、Trace（即图状态） | LangGraph checkpointer 持久化，Redis 提供跨实例共享 |
| **L2 长期记忆** | 情景记忆（历次运行的会话摘要）+ 语义 / 画像（跨会话沉淀的事实与偏好） | 向量检索 |

记忆是**活的**，不只是报告里的展示：

- 运行开始时，按相关度检索 L2，把命中的记忆注入 planner 上下文；
- 运行结束后，把本次结论与关键事实写回、固化到 L2；
- Agent 可通过 `recall` 工具主动检索记忆、通过 `remember` 工具主动记账；
- **边界很清楚：记忆只提供背景与偏好，永远不作为 JD / 简历的判断证据**，不污染判定。

### 5. 知识库：检索增强的另一条腿

记忆存的是「这次和以前发生过什么」，知识库存的是「从资料里能检索到什么」。

| 层 | 内容 | 来源 |
|----|------|------|
| L1 静态知识 | 稳定的方法论、词典、业务规则 | `knowledge/` |
| L2 半静态知识 | 会随文件更新的 JD、项目描述、简历 | `input/` |
| L3 动态知识 | 持续变化的历史报告、运行结果 | `output/` |

增量索引用文件 SHA-256 判断是否变化；没配 Embedding Key 时用确定性的本地哈希嵌入，不联网；配置后使用通义 `text-embedding-v4` 走向量检索。

### 6. 工具层：一个契约，处处复用

每个工具都是一个 `Tool` 子类、统一返回 `ToolResult`，注册在 `jd_agent/tools/` 的注册表里。
命令行、Web 后端与 Agent 循环共用同一份实现，新增工具只需实现 `run()` 并登记进注册表。

### 7. 全模态输入

JD 与简历各自支持 **文本 / 文件 / 图片 / 网址** 四类输入，另有项目描述、`--answer` 补答与 Web 上传：
文本直接读；文件走文件解析工具；图片交给 Qwen-VL 逐字转写；网址抓一次 HTML 再转 Markdown。

---

## 快速开始

### 在 PyCharm 里

1. `File > Open` 选择项目目录；
2. 解释器选任意 Python 3.9+，运行 `pip install -r requirements.txt`（python-dotenv + langgraph + fastapi/uvicorn）；
3. 右键 `main.py` → **Run 'main'**（不指定 `--jd` 时会把 `input/jd` 下所有岗位各跑一遍）；
4. 结果写在 `output/`：**目录里只有 JSON**，单个岗位是 `resume_decision.json`，多个岗位按岗位分文件；需要 Markdown / HTML 再加 `--formats md,html` 导出。

### 命令行

```bash
cd path\to\AI_Job_Research_Agent
python main.py                                         # 不指定 --jd：跑完 input/jd 下所有文件的全部岗位
python main.py --jd input/jd/xx.md                     # 只跑这个文件（文件里多个岗位时默认只跑第一个）
python main.py --jd input/jd/xx.md --jd-title 岗位二    # 指定这个文件里的哪个岗位
python main.py --answer "这个项目的结果是……"            # 回答 Agent 提出的那一个问题
python main.py --project input/profile/资料不足项目测试简历1.md   # 第二个输入可以是任何 md

python main.py --build-resume                          # 简历排版 → output/resume.json（要 md/html 加 --resume-format md,html）
python main.py --build-resume --resume-style structure # 换风格（classic / structure / accent）

python main.py --jd-agent --jd-text "岗位职责：……"       # JD Agent：JD（文本/图片/网址）→ 只打印，要导出加 --jd-out
python main.py --jd-agent --jd-url "https://…/job/1"    # 网址：抓一次 HTML → Markdown
python main.py --jd-agent --jd-image assets/jd.png --jd-trace
python main.py --resume-agent --cv-file input/profile/xx.md --cv-trace   # 简历 → 事实条目 + 证据等级
python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md
python main.py --polish-agent --jd-file … --cv-file … --polish-trace

python main.py --llm                                   # 启用 LLM 策略（text 层，需 .env）
python main.py --strategy llm                          # v4：有界 plan → act → observe + verify（需 .env）
python main.py --llm --strategy rule                   # 兼容旧装饰层：规则结案后补草稿 / 润色
python main.py --vision                                # 启用图片层：Qwen-VL 把图片读成文字事实（需 .env）
python main.py --enable-memory --session-id demo       # 启用 L0/L1/L2 记忆，L2 写入 data/memory/
python main.py --enable-knowledge --knowledge-tiers L1,L2   # 启用知识检索
python main.py --index-only                            # 只建知识索引后退出

.venv\Scripts\python.exe -m streamlit run streamlit_app.py   # 可选：Streamlit 前端
```

**退出码**：`0` 已给出判断（Stop）｜`3` 需要你回答一个问题（Ask）｜`2` 输入有误。

### Web 前端（Vue3 + FastAPI）

```bash
# 一条命令同时起后端与前端（PowerShell，自动用 .venv；Ctrl+C 一起停）
powershell -ExecutionPolicy Bypass -File scripts/dev.ps1
# 前端 http://127.0.0.1:5173，后端 http://127.0.0.1:8000

# 或分开起：
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
cd frontend; npm install; npm run dev              # 首次要先 npm install
```

前端五页：输入 / 进度 / 报告 / 简历润色 / 历史；后端接口清单见 [`BACKEND.md`](BACKEND.md)。

---

## 输出与 Trace

| 产物 | 说明 |
|------|------|
| `output/resume_decision.json` | 主链判断报告：结论、理由、风险、`StopReason`、完整 Trace。**`output/` 只收 JSON** |
| `output/resume_decision_<jd_id>.json` | 批量模式：多个岗位按岗位分文件 |
| `output/resume.json` | `--build-resume` 排好版的简历；要 md/html 加 `--resume-format md,html` |
| JD / Resume / Polish Agent | 默认只把结果打印到终端；要导出加 `--jd-out` / `--cv-out` / `--polish-out` |

Trace 同时打印在终端，**每个字段都能回查原文**（`文件::L行号`）。

---

## 目录结构

```text
AI_Job_Research_Agent/
├── main.py                  # 入口（PyCharm 直接 Run）
├── streamlit_app.py         # 可选：Streamlit 前端
├── api/main.py              # FastAPI 后端入口
├── BACKEND.md               # 后端接口清单
├── frontend/                # Vue3 + Vite 前端（五页）
├── scripts/dev.ps1|dev.sh   # 一条命令同时起后端与前端
├── knowledge/               # L1 静态知识（方法论 / 词典 / 业务规则）
├── data/                    # 运行时数据：记忆 / 知识索引 / 上传临时文件（已忽略）
├── input/
│   ├── jd/                  # 岗位描述
│   ├── project/             # 项目描述
│   └── profile/             # 简历 md
├── output/                  # 报告输出
├── tests/                   # 离线测试集（注入假 client，不联网）
└── jd_agent/
    ├── core/                # 数据结构 / 设置 / LLM 客户端
    ├── domain/              # 能力词典 / JD 解析 / 事实分级 / 证据匹配
    ├── agents/              # Agent 循环 + JD / Resume / Polish 三张图 + 渲染
    ├── tools/               # 工具区：一个工具 = Tool 子类 + run()
    ├── memory/              # L0 / L1 / L2 记忆实现
    ├── knowledge/           # 知识索引与向量检索
    ├── services/            # 四种来源识别、图片转写
    └── api/                 # FastAPI 后端：routers / services / stores
```

---

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：JD 切分与层级去重、项目事实定级（含背景降级与否定识别）、严格 / 放宽两种匹配、Trace 四要素完整性、
Ask / Answer / Stop 三条路径、CLI 退出码与三种输出格式、批量岗位与简历排版、三条 Agent 链的节点顺序与降级路径。

大模型层与视觉层的用例**全程不联网**（注入假 client，一次请求都不发），验证「LLM 不做结论」、失败可降级、
调用纪律（超时 / 重试 / 调用上限）与「缺 Key 时一次请求都不发」。

---

## 可信边界

- **判断的是「能不能撑住这条要求」，不是「你水平多高」**：项目描述写得实，判断才准，这也是它先问你问题的原因。
- **每条理由、每个风险都带原文行号**，结论可追溯；阈值是为了可解释和可调，不是行业标准。
- **图片解析结果建议人工复核**：模型可能看错刻度、漏读单位，事实本身仍要对着原图核一遍。
- **LLM 生成内容不是判定依据**：写法草稿、追问预演与润色只用来准备面试与改简历；「值不值得写」这个结论、`StopReason` 与每一步的 `Decision` 始终来自规则。
- **记忆只提供背景，不作证据**：跨会话的事实与偏好会注入规划上下文，但不会进入 JD / 简历的证据池。

---

## 后续演进

- 多项目横向排序，直接回答「先写哪个」；
- 固化任务与用户画像，让评估集与个人偏好沉淀得更稳定；
- 工具层对接 MCP 协议，把现有工具开放给外部 Agent 编排。
