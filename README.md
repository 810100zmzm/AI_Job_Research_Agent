# AI 求职尽调 Agent · 这个项目值不值得写进简历？

做过几个 AI 项目，却说不清哪个真正适合写进简历去投 AI 工程师岗；更担心的是，面试官顺着简历往下问，自己答不上来。

这个 Agent 只做一件事：

> 读 **一份 JD** + **一个项目描述** → 抽出岗位真正需要的能力 → 到项目里逐条找能支撑这些能力的证据
> → 给出**要不要写进简历**的判断（2~3 条理由 + 1~2 个风险），并留下可以逐条检查的运行 Trace。

它不猜：资料不够时停下来问你**一个**最关键的问题；证据够了才下结论，并写明 `StopReason`。

**纯离线** —— 规则 + 能力词典（`jd_agent/domain/lexicon.py`），不需要 API Key：不加开关就一个请求都不发，也不读 `.env`。
需要「建议写法草稿」和「图片里的证据」时再加 `--llm` / `--vision`（见下文「可选：接大模型」）。
JD Agent 也一样：`--jd-text` / `--jd-file` 两条分支不联网、不读 `.env`，只有 `--jd-url`（抓一次 HTTP）与
`--jd-image`（Qwen-VL）会发请求。
Resume Agent 同理：`--cv-text` / `--cv-file` 两条分支不联网、不读 `.env`，只有 `--cv-url` 与 `--cv-image` 会发请求。

v1.1 顺手补了两件事：**批量**（不指定 `--jd` 就跑完 `input/jd` 下所有文件的**全部岗位**，一个岗位一份报告）
和**简历排版**（`--build-resume` 把手写的简历排成简约大方的 md / html）；想要网页版还有 Streamlit 前端。

v2.0 把简历排版收进了**工具区 `jd_agent/tools/`**：每个工具都是一个 `Tool` 子类、统一返回 `ToolResult`，
命令行与前端共用同一份实现，不再各写一套；排版另加**三套风格**（`--resume-style classic|structure|accent`）。
默认那套（classic）保持通用、克制的排版，不换风格就不用改使用习惯。

v2.1 加了 **JD Agent（LangGraph）**：把 JD（**文本 / 图片 / 网址**）转成统一模板的 Markdown —— 文本直接读、
截图交给 Qwen-VL 逐字转录、网址抓一次 HTML，再交给工具区的**文件解析 Tool** 与**JD 结构化 Tool** 规整成
「岗位标题 + 元信息 + 岗位职责 / 任职要求 / 加分项」。它走的是同一套 Agent Loop（四要素 Trace、四种 Decision）。

v2.2 加了 **Resume Agent（LangGraph）**：把简历（**文本 / 文件 / 图片 / 网址**）拆成一条条事实条目，逐条判证据等级。
它的重点是 **先挡伪装，再判等级** —— 先用「否定 / 背景识别 Tool」把「写了但没做过 / 只是背景」的条目挑出来
（「了解向量数据库，**没做过** Docker 部署」里藏着「部署」这个动词，不先挡就会被当成功夫），
再用「证据等级判定 Tool」把剩下的条目定级（有结果 / 有动作 / 仅提及）。被挡掉的条目连带原文与行号一起列出来，
可以逐条核对挡得对不对；**否定与背景放宽口径也不放**。同样走四要素 Trace 与四种 Decision。

v2.3 加了 **Polish Agent（LangGraph）**：把前两条链合起来 —— 基于**已核验事实 + JD 要求**生成
**建议写法 / 追问预演 / 润色**（要 JD 与简历各给一路）。这一层最要紧的分工是
**工具负责「模型能看到什么」，模型只负责「怎么说」**：模型读不到 JD / 简历全文，只读一份由规则生成的
写作简报（要求 + 已核验事实 + 缺口与风险 + 被挡掉的内容）；「写了但没做过」只告诉它「别写」。
新增的**能力词典匹配 Tool（`cap-match`）**负责把「JD 要什么」与「简历里有什么」摆到同一张表上。

v2.4 加了 **分层记忆与知识库**：记忆分 L0 会话 / L1 短时 / L2 长时，知识分 L1 静态 / L2 半静态 / L3 动态。
默认全部离线：L1 用进程内字典，L2 用项目 `data/memory/long_term.jsonl`，知识索引用 `data/knowledge/index.jsonl`；
配好 MongoDB / Qdrant / 通义 `text-embedding-v4` 后可切换后端。检索结果只进入 `state.context` 作为补充，
**不改规则结论、理由、风险与 Decision**。所有运行时数据只允许落在项目同一个磁盘的 `data/` 下。

代码按职责放在 `jd_agent/core`、`jd_agent/domain`、`jd_agent/services`、`jd_agent/agents` 与
`jd_agent/tools`；根目录只保留命令行入口。目录与依赖约束见 [`jd_agent/README.md`](jd_agent/README.md)。

---

## 快速开始

### 在 PyCharm 里

1. `File > Open` → 选择项目目录 `D:\codexAbout\projects\AI_Job_Research_Agent`
2. 解释器选任意 Python 3.9+，然后 `pip install -r requirements.txt`（python-dotenv + langgraph）
3. 右键 `main.py` → **Run 'main'**（不指定 `--jd` 时会把 `input/jd` 下所有岗位各跑一遍）
4. 结果写在 `output/`：单个岗位是 `resume_decision.md｜json｜html`，多个岗位按岗位分文件

### 命令行

```bash
cd D:\codexAbout\projects\AI_Job_Research_Agent
python main.py                                     # 不指定 --jd：跑完 input/jd 下所有文件的全部岗位
python main.py --jd input/jd/xx.md                 # 只跑这个文件（文件里多个岗位时默认只跑第一个）
python main.py --jd input/jd/xx.md --jd-title 岗位二   # 指定这个文件里的哪个岗位
python main.py --answer "这个项目的结果是……"        # 回答 Agent 提出的那一个问题
python main.py --project input/profile/资料不足项目测试简历1.md   # 第二个输入可以是任何 md，看 Ask 分支长什么样
python main.py --build-resume                      # 简历排版 → output/resume.md + output/resume.html
python main.py --build-resume --resume-style structure # 换风格（classic / structure / accent）→ resume-structure.*
python main.py --jd-agent --jd-text "岗位职责：……"   # JD Agent：JD（文本/图片/网址）→ output/jd.md
python main.py --jd-agent --jd-url "https://…/job/1"   # 网址：抓一次 HTML → Markdown
python main.py --jd-agent --jd-image assets/jd.png --jd-trace   # 图片走 Qwen-VL，附带完整 Trace
python main.py --resume-agent --cv-file input/profile/xx.md   # Resume Agent：简历 → 事实条目 + 证据等级清单
python main.py --resume-agent --cv-image assets/cv.png --cv-trace   # 简历截图走 Qwen-VL，附带完整 Trace
python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md   # Polish Agent：建议写法 / 追问预演 / 润色
python main.py --polish-agent --jd-file … --cv-file … --polish-trace   # Polish Agent 附带完整 Trace
python main.py --llm                               # 可选：报告生成后补写法草稿 / 润色（需 .env）
python main.py --vision                            # 可选：Qwen-VL 把图片读成文字事实（需 .env）
python main.py --enable-memory --session-id demo   # 开 L0/L1/L2；L2 默认写 data/memory/long_term.jsonl
python main.py --index-only                        # 按 L1/L2/L3 增量建立知识索引后退出
python main.py --enable-knowledge --knowledge-tiers L1,L2   # 检索静态 / 半静态知识

.venv\Scripts\python.exe -m streamlit run streamlit_app.py      # 可选：网页版前端
```

**要分析哪份 md**：JD 用 `--jd`（不给就是 `input/jd` 下全部岗位），第二个输入用 `--project`。
第二个输入不限于 `input/project/`，你自己的简历 md 也能直接喂进去，例如：

```bash
python main.py --project input/profile/资料不足项目测试简历1.md
```

**退出码**：`0` 已给出判断｜`3` 需要你回答一个问题｜`2` 输入有误。

---

## 分层记忆与知识库

记忆存的是“这次和以前发生过什么”，知识库存的是“从资料里能检索到什么”。两者默认关闭，显式开关后才初始化。

| 层 | 用途 | MVP 实现 | 可选后端 |
|---|---|---|---|
| L0 会话记忆 | 当前会话的精确消息与状态 | 进程内 `SessionMemory`；另保留 LangGraph checkpoint 适配入口 | 后续可接持久化 checkpointer |
| L1 短时记忆 | 有 TTL、有容量上限的工作记忆 | 进程内字典；不占用磁盘 | 换 Redis 只需改 `jd_agent/memory/short_term.py` |
| L2 长时记忆 | 跨会话保留的重要决策与用户偏好 | `data/memory/long_term.jsonl` | MongoDB（`MONGODB_URI`） |
| L1 静态知识 | 稳定的方法论、词典、业务规则 | 读取 `knowledge/` | Qdrant / 本地 JSONL 索引 |
| L2 半静态知识 | JD、项目描述、简历等会随文件更新 | 读取 `input/jd`、`input/project`、`input/profile` | Qdrant / 本地 JSONL 索引 |
| L3 动态知识 | 历史报告、运行结果等会持续变化的内容 | 读取 `output/` | Qdrant / 本地 JSONL 索引 |

### 索引与检索

```bash
python main.py --index-only                         # 只增量建索引，不跑 JD 分析
python main.py --index-knowledge                    # 先增量建索引，再继续常规分析
python main.py --enable-knowledge                   # 只检索已有索引
python main.py --enable-knowledge --knowledge-tiers L1,L3
```

增量索引用文件 SHA-256 判断是否变化；文件变化时先删除该文档旧切片，再写入新切片。没配置 `DASHSCOPE_API_KEY` /
`QWEN_API_KEY` 时自动用确定性的本地哈希嵌入，不联网；配置后默认使用通义 `text-embedding-v4`。

### 数据位置与位置约束

默认数据目录是项目根目录下的 `data/`，已被 `.gitignore` 忽略。Windows 上如果显式传入另一个磁盘的
`--data-dir` 或 `DATA_DIR`，配置解析会直接拒绝，避免把记忆或索引写到 C 盘。

### 可选依赖

```bash
pip install pymongo       # L2 长时记忆使用 MongoDB 时需要
pip install qdrant-client # 知识索引用 Qdrant 时需要
```

配置项见 [`.env.example`](.env.example)。重要的一个接口边界：Agent 只依赖
`jd_agent/memory/interfaces.py` 里的 `ShortTermMemory`，因此 MVP 的内存实现换成 Redis 时，`agents/`、`cli.py`
与 Streamlit 都不需要改。

## 一次真实运行

```text
[1] Action      : ReadJD
    Observation : 读取 3份校招_秋招_日常实习 AI岗位描述.md（117 行），识别到 3 个岗位，
                  锁定「岗位二：AI大模型应用开发实习生」；该岗位 13 行与能力要求相关
    State Update: state.jd = 「岗位二…」（未标注公司）
    Decision    : Continue —— JD 可读，已定位到一个具体岗位

[2] Action      : ReadProject
    Observation : 读取 示例项目1-AI周报助手.md（34 行），解析出 15 条事实：
                  有结果 3 / 有动作 7 / 仅提及 5；能看出「这是我做的」2 条
    State Update: state.project = 15 条事实
    Decision    : Continue —— 项目描述已拆成可核验的事实条目

[3] Action      : ExtractRequirements
    Observation : 16 项能力里，4 项属于门槛 / 素质类（学历、届别、实习时长、学习能力）——
                  项目无法证明，移出对比；剩下 12 项进入检索，其中必备 7 项
    State Update: state.core_requirements = 12 项，state.skipped = 4 项
    Decision    : Continue —— 已得到「项目能证明什么」的能力清单

[4] Action      : RetrieveEvidence
    Observation : 严格匹配（只认「有动作 / 有结果」）：7/12 项命中，其中带量化结果 3 项；
                  无证据：文档撰写、用户反馈与迭代、前沿跟进、RAG 与知识库应用、Docker 与部署
    State Update: state.matches = 12 条（命中 7）
    Decision    : Adjust —— 5 项在严格规则下没证据，可能只是写成了技术栈 → 放宽规则再查一轮

[5] Action      : AdjustEvidence
    Observation : 放宽后（允许「仅提及」级计入弱证据）新增 0 项弱证据；仍有 5 项完全找不到
    State Update: state.matches = 有结果 3 / 有动作 4 / 仅提及 0 / 无证据 5
    Decision    : Continue —— 证据已尽量取全，可以判断资料够不够用

[6] Action      : Judge
    Observation : 核心覆盖 7/12（58%），结果级证据 3 条，仍缺 5 项
    State Update: state.verdict = 「值得写，但必须先改写」；理由 3 条、风险 2 条
    Decision    : Stop —— 12 项核心要求已逐条核对完，证据足以支撑这个结论，继续检索不会产生新信息。
```

结论区（同一份报告里）：

```text
结论：值得写，但必须先改写
  只撑得住 7/12 项核心要求（覆盖 58%），还差 5 项；可以写，但必须限定在你真做过、讲得清的部分。

为什么（3 条）：
  1. JD 的任职要求提到「接口 / Web 后端开发」：熟练使用Python，了解FastAPI…（input\jd\…::L63）
     → 项目里有对应证据：用 FastAPI 把摘要流程封装成 HTTP 接口…（input\project\…::L16，有动作）
  2. …
风险（2 条）：
  - [证据缺口] 「RAG 与知识库应用」是 JD 的任职要求（…::L65），这段项目里没有对应证据…
  - [追问风险] 「接口 / Web 后端开发」只有动作、没有结果（…::L15），面试官会追问指标…
StopReason：12 项核心要求已逐条核对完（覆盖 58%、结果级证据 3 条、缺口 5 项）…
```

资料不足时会是这样：

```text
[6] Action      : CheckSufficiency
    Observation : 资料不足：4 条事实里没有一条能看出「我做了什么」，全是背景介绍与技术栈罗列
    State Update: state.sufficiency = insufficient；
                  state.question =「这个项目里你个人具体做了什么？哪几个模块是你独立完成的？」
    Decision    : Ask —— 项目描述里看不出你的贡献，替你猜会直接把结论带偏
```

---

## Agent Loop：每一步都留四样东西

每完成一步，Trace 里就落一条记录，四要素缺一不可：

| 字段 | 含义 |
|------|------|
| **Action** | 这一步做了什么（读 JD / 读项目 / 图片取证〔可选〕/ 抽能力 / 检索证据 / 放宽重查 / 判断资料够不够 / 下结论；`--llm` 时还有报告生成后的三步 `GenerateDraft / RehearseInterview / PolishReport` 与收尾 `Finish`） |
| **Observation** | 实际看到了什么：数字、文件名、行号，而不是结论 |
| **State Update** | 状态被改成了什么（`state.matches = …`） |
| **Decision** | 下一步只能选这四种之一，并写明理由 |

**Decision 只有四种**：

| Decision | 什么时候用 |
|----------|-----------|
| `Continue` | 这一步的信息够了，按计划继续下一步 |
| `Adjust` | 搜索结果不理想，换一种检索口径再查一次（例如先严格后放宽） |
| `Ask` | 缺的是只有你才知道的事实，停下来问你**一个**问题 |
| `Stop` | 证据足够支撑结论，输出判断并写明 `StopReason`；带 `--llm` 时由最后一步 `Finish` 落这条 `Stop` |

### 什么时候它会停下来问你

满足任一条就 `Ask`（只问排在最前面的那一个问题，问完还不足就给保守结论，不无限追问）：

1. 项目描述里看不出「我做了什么」（全是背景介绍和技术栈罗列）→ 问你的个人贡献；
2. 项目与岗位核心要求零交集 → 问最接近的那一项你到底做没做过；
3. 事实少于 3 条 → 请你补 3 行（负责哪块 / 什么方法 / 结果如何）；
4. 没有结果级证据且覆盖 < 50% → 问项目的量化结果。

回答方式：`python main.py --answer "…"`（回答会作为新材料并入证据池，重新检索一轮），
或直接把回答补进项目描述文件里再跑一次。

---

## 判定规则（全部写死，可直接检查）

### 1. 证据等级：项目里的每条事实都会被定级

| 等级 | 判定条件 |
|------|---------|
| ✅ **有结果** | 有量化指标（`准确率 85%`、`从 62% 提升到 88%`），或有交付动作（`已开源 / 已上线 / 已部署`）且能看出是你做的 |
| 🟠 **有动作** | 有 `搭建 / 实现 / 调试 / 调优 / 修复` 这类动词，但没写结果 |
| 🟡 **仅提及** | 只出现在技术栈罗列里，或只有「用过 / 了解」 |
| ❌ **无证据** | 项目里找不到 |

两个刻意的设计：

- **背景段落降级**：`## 背景` 里的「人工整理要 3 小时」是问题描述，不是你的交付证据，一律算「仅提及」；
- **否定句不算证据**：`没做过模型微调`、`未接触过向量库` 会被单独记下来，既不能当证据，还会变成风险提示。

### 2. 岗位要求：只留项目能证明的那部分

- 学历、届别、可实习时长、通用素质（学习能力 / 主动性）会被**移出对比**（项目证明不了这些）；
- 同一项能力被 JD 多行提到时，只保留层级最高的一条（任职要求 1.0 > 岗位职责 0.8 > 加分项 0.5）。

### 3. 结论三档

| 结论 | 条件 |
|------|------|
| ✅ 值得写 | 核心要求覆盖 ≥ 60%，且有量化结果，且必备项无缺口 |
| 🟠 值得写，但必须先改写 | 覆盖 35%~60%，或必备项有缺口 |
| ❌ 暂不建议写 | 覆盖 < 35%，或没有任何动作级证据 |

### 4. 理由与风险怎么来的

- **理由 2~3 条**：按 `层级权重 × 证据等级` 排序，优先让每条理由来自**不同的 JD 行和不同的项目证据**；证据不够时用「仅提及」级补齐并明确标注「（弱证据）」；
- **风险 1~2 条**，按优先级取：必备项没证据 → 项目里写了没做过 → 只有动作没有结果 → 只有技术栈级证据；
- **建议写法**：只用已核验的证据拼一版草稿，不加形容词、不编数字。

---

## 输入怎么写

### JD（`input/jd/*.md`）

```markdown
## 岗位二：AI大模型应用开发实习生

**岗位职责**
- 协助团队进行大模型应用开发与调试，参与智能问答、知识库RAG功能开发。

**任职要求**
- 熟练使用Python，了解FastAPI/Flask等基础Web开发框架。
- 熟悉大模型基础原理，了解RAG、向量数据库、Prompt工程基础内容。

**加分项**
- 熟悉Docker基础部署流程
```

章节名支持 `任职要求 / 岗位要求 / 岗位职责 / 加分项` 等常见写法（`##`、`**加粗**`、`冒号结尾` 都认）。
一个文件里放多个岗位也行：**不指定 `--jd` 时会把每个岗位拆开各跑一遍**（见下文「批量」）；
用 `--jd` 指定单个文件时保持 v1.0 的行为——默认只跑第一个，要切换就用 `--jd-title 岗位二`。

### 项目描述（`input/project/*.md`，也可以是任何 md）

第二个输入，命令行里对应 `--project`：可以是 `input/project/` 下的项目描述，也可以是 `input/profile/` 下
你自己写的简历 md——程序不关心文件放在哪，只关心里面有没有**可核验的事实**。
按四段写，判断质量几乎完全取决于这四段写得实不实：

```markdown
# 项目：AI 周报助手

## 背景与目标
- 每周整理 AI 资讯要 3 小时。

## 我做了什么          ← 最重要：写清「我独立 / 负责」的部分
- 独立完成采集、摘要、接口封装与页面联调，代码已开源到 GitHub。
- 调试阶段定位过长文截断问题，改成滑动窗口分块后修复。

## 技术栈
- Python、FastAPI、Chroma、Git

## 结果                ← 有数字就给数字
- 摘要可用率从 62% 提升到 88%（人工抽检 100 条）。
```

## JD Agent：把 JD（文本 / 图片 / 网址）转成 Markdown

主线回答的是「写不写」，前提是**手上先有一份结构清楚的 JD**。可实际情况往往是：招聘 App 里复制出来的一坨文字、
一张岗位截图、一个只能看不能导出的网页。这一层就干这个 —— 用 **LangGraph** 编排，把三种输入都转成
**统一模板的 Markdown**，转完可以直接喂回主线：

```bash
python main.py --jd-agent --jd-text "岗位职责：……"         # 文本：直接读取（不联网、不调模型）
python main.py --jd-agent --jd-file input/jd/xx.md          # 文件：走文件解析 Tool（md/txt/html/json/yaml）
python main.py --jd-agent --jd-image assets/jd.png          # 图片：Qwen-VL 逐字转录（需 .env 里的 key）
python main.py --jd-agent --jd-url "https://…/job/1"        # 网址：抓一次 HTML → Markdown
python main.py --jd-agent --jd-url "https://…" --jd-trace   # 附带完整 Trace（…trace.md）
python main.py --jd-url "https://…" --jd-out output/my_jd.md --jd-name "AI 实习岗"   # 指定输出与标题

python main.py --jd output/jd.md --project input/project/示例项目1-AI周报助手.md   # 转完直接跑主线
```

产出长这样（`output/jd.md`）：

```markdown
# AI 大模型应用开发实习生

> 由 JD Agent（LangGraph）转换｜来源：jobs.example.com｜生成时间：2026-09-23 10:20｜章节由规则识别，一字未改

**公司**：某某科技
**工作地点**：北京

**任职要求**
- 熟练使用 Python，了解 FastAPI 等 Web 框架。
- 熟悉大模型基础原理，了解 RAG 与向量数据库。
```

### 图怎么走（`jd_agent/agents/jd_graph.py`）

一张 LangGraph 状态图；节点名就是 Trace 里的 `Action`：

| 节点 | 干什么 | 这一步的 Decision |
|------|--------|-------------------|
| `detect` | 识别输入类型，把读不了的来源挑出来并说明原因 | `Continue` |
| `read_text` | 文本直接读取：不联网、不调模型 | `Continue` |
| `parse_file` | 文件交给工具区的**文件解析 Tool**（纯规则） | `Continue` |
| `ocr_image` | 图片交给 **Qwen-VL** 逐字转成 Markdown | `Continue`（失败只降级） |
| `fetch_url` | 网址抓一次 HTML，转成 Markdown（严格口径） | `Continue` |
| `relax` | 严格口径剩下的正文太少：换**放宽口径**从同一份内容重抽（不再发请求） | `Adjust` |
| `structure` | 交给工具区的 **JD 结构化 Tool**：切岗位 + 分章节 + 出模板 | `Continue` |
| `finalize` | 写出 Markdown，写明 `StopReason` | `Stop` |
| `ask` | 内容太少：停下来问**一个**问题，不替用户编 JD | `Ask` |

三种输入各走各的分支，队列里的来源都取完了才进 `structure`；「来源取不到」只是降级 —— 记一条备注继续跑剩下的，
最后如实写进 `StopReason`，而不是整条链路崩掉。

### 复用了哪两个工具

| 工具 | slug | 干什么 | 边界 |
|------|------|--------|------|
| 文件解析 Tool | `jd-file` | 本地 JD 文件（md / txt / html / json / yaml）→ Markdown 草稿 | 纯规则：**不联网、不读 `.env`、不调模型**；图片交给图片分支 |
| JD 结构化 Tool | `jd-struct` | JD 原文 → 统一模板（岗位标题 + 元信息 + 岗位职责 / 任职要求 / 加分项） | 纯规则；认不出的章节原样保留，不往「任职要求」里塞 |

两个工具的命令行与前端共用同一份实现，`--jd-agent` 走的也是同一份，不存在两套口径。

### 三种输入分别会做什么

| 输入 | 会联网吗 | 会调模型吗 | 说明 |
|------|----------|------------|------|
| `--jd-text` | 否 | 否 | 整段就是一个网址时自动改走网址分支 |
| `--jd-file` | 否 | 否 | 后缀决定怎么解析；文件不存在 / 后缀不认识 → 退出码 2 并说明原因 |
| `--jd-image` | 是（1 次模型调用 / 张） | 是（Qwen-VL） | 逐字转录，看不清的字写 `[看不清]`；没配 key 就跳过这一路 |
| `--jd-url` | 是（1 次 HTTP GET / 个） | 否 | 只抓你给的那个页面：标准库 urllib，带浏览器 UA、gzip、3 MB 截断 |

### 两道阈值（写死在 `jd_agent/agents/jd_graph.py`，可直接检查）

| 常量 | 值 | 触发什么 |
|------|----|----------|
| `MIN_URL_LINES` | 5 | 严格口径下有效行少于 5 行 → `Adjust`：换放宽口径重抽（两个口径都不改字，只决定留哪些行） |
| `MIN_JD_BULLETS` | 3 | 转出来少于 3 条要点 → `Ask`：问一句，而不是硬出一份 JD |

退出码与主线一致：`0` 已写出 Markdown（`Stop`）｜`3` 需要你回答一个问题（`Ask`）｜`2` 输入有误。

---

## Resume Agent：把简历拆成事实条目，逐条判证据等级

主线要的是「项目证据」，Resume Agent 要的是**一份可以直接核对的简历事实清单**：把简历（文本 / 文件 / 图片 / 网址）
拆成一条条事实，写明原文出处，并逐条给出三个答案 —— **算不算证据、是哪一档、为什么**。
转出来的 Markdown 可以再当 `--project` 的输入喂回主线。同样用 **LangGraph** 编排：

```bash
python main.py --resume-agent --cv-text "李小明…"                    # 文本：直接读取（不联网、不调模型）
python main.py --resume-agent --cv-file input/profile/xx.md          # 文件：复用文件解析 Tool（md/txt/html/json/yaml）
python main.py --resume-agent --cv-image assets/cv.png               # 图片：Qwen-VL 逐字转录（需 .env 里的 key）
python main.py --resume-agent --cv-url "https://…/cv"                # 网址：抓一次 HTML → Markdown
python main.py --resume-agent --cv-file … --cv-trace                 # 附带完整 Trace（…trace.md）
python main.py --resume-agent --cv-file … --cv-title 我的简历 --cv-out output/my_facts.md
```

产出长这样（`output/resume_facts.md`）：

```markdown
# 李小明 · 简历事实清单

> 由 Resume Agent（LangGraph）生成｜来源：input/profile/xx.md｜事实与等级由规则判定，一字未改

**姓名**：李小明
**求职意向**：AI 大模型应用开发实习生

**可当证据的事实**：6 条（硬证据 2 / 弱证据 4）｜**被挡掉**：2 条（否定 2 / 背景 0）

**一句话结论**：6 条事实能当证据（硬证据 2：有结果 1 / 有动作 1；弱证据 4：仅提及 4）；本次按严格口径计入 2 条；2 条被挡掉（否定 2 / 背景 0），它们都不能算「已具备的证据」。

## 事实清单

### 项目经历

| # | 事实 | 等级 | 命中能力 | 依据 | 出处 |
| --- | --- | --- | --- | --- | --- |

## 不能当证据的内容（挡住「写了但没做过」）

| # | 原文 | 判定 | 为什么不算证据 | 出处 |
```

### 判定顺序：先挡伪装，再判等级

这一层的重点不是「谁等级高」，而是**别把「写了但没做过 / 只是背景」当成已具备的证据**。四道关按顺序过，
顺序不能反 —— 反了就会把「没做过 Docker 部署」里的「部署」当成功夫：

| 顺序 | 挡什么 | 认哪些写法 | 例子 |
|------|--------|------------|------|
| ① 否定 | 反向证据 | `未接触 / 没做过 / 没经验 / 尚未 / 没有任何…实习经历`；关键词**前后都看**，标点即边界 | 「了解向量数据库，**没做过** Docker 部署」→ 整行不算证据 |
| ② 背景 / 计划 | 不是「我做过什么」 | 整节叫「项目简介 / 课题背景」，或行里写着「本项目**旨在**…」「**计划**学习 Go」 | 「**计划**学习 Go」→ 只是打算，不等于已经具备 |
| ③ 课程封顶 | 上过课 ≠ 做过事 | `修读 / 选修 / 培训 / 自学`，且这一行没有任何指标 | 「培训期间完成了数据标注」→ 封顶「仅提及」（写出指标就不封） |
| ④ 判等级 | 有结果 > 有动作 > 仅提及 | 量化指标 / 交付词（开源、上线）/ 动作词（搭建、实现）/ 归属词（我、独立、负责）；另加简历口径：**荣誉与名次**算「有结果」 | 「准确率从 72% 提升到 85%」→ 有结果 |

两个口径：**严格**只认「有动作 / 有结果」；**放宽**（`Adjust`）才把「仅提及」也算成弱证据。
但 **否定与背景永远不放宽** —— 「没做过」写多少遍都不会变成「做过」，这正是这一层存在的理由。

被挡掉的条目不会被丢掉，而是连同原文与行号一起列进「不能当证据的内容」，一眼就能核对它挡得对不对。

### 图怎么走（`jd_agent/agents/resume_graph.py`）

一张 LangGraph 状态图；节点名就是 Trace 里的 `Action`：

| 节点 | 干什么 | 这一步的 Decision |
|------|--------|-------------------|
| `detect` | 识别输入类型，把读不了的来源挑出来并说明原因 | `Continue` |
| `read_text` | 文本直接读取：不联网、不调模型 | `Continue` |
| `parse_file` | 文件交给工具区的**文件解析 Tool**（纯规则） | `Continue` |
| `ocr_image` | 图片交给 **Qwen-VL** 逐字转成 Markdown | `Continue`（失败只降级） |
| `fetch_url` | 网址抓一次 HTML，转成 Markdown（简历页不套 JD 那套 UI 噪声词表） | `Continue` |
| `structure` | 交给**简历结构化 Tool**：切章节 + 认抬头信息 + 抽事实条目（带原文行号） | `Continue` |
| `screen` | 交给**否定 / 背景识别 Tool**：先把「没做过 / 只是背景」的条目挡出来 | `Continue` |
| `evidence` | 交给**证据等级判定 Tool**：剩下的条目才判「有结果 / 有动作 / 仅提及」 | `Continue` |
| `relax` | 严格口径下一条硬证据都没有：换**放宽口径**重判（不再读文件、不再发请求） | `Adjust` |
| `finalize` | 写出 Markdown，写明 `StopReason` | `Stop` |
| `ask` | 内容太少、或一条能当证据的都没有：停下来问**一个**问题，不替你编经历 | `Ask` |

四种输入各走各的分支，队列里的来源都取完了才进 `structure`；「来源取不到」只是降级 —— 记一条备注继续跑剩下的，
最后如实写进 `StopReason`，而不是整条链路崩掉。

### 四个工具（三个新增 + 一个复用）

| 工具 | slug | 干什么 | 边界 |
|------|------|--------|------|
| 文件解析 Tool（复用） | `jd-file` | 本地简历文件（md / txt / html / json / yaml）→ Markdown 草稿 | 纯规则：**不联网、不读 `.env`、不调模型** |
| 简历结构化 Tool | `resume-struct` | 简历原文 → 章节 + 抬头信息 + 事实条目（带原文行号） | 纯规则；**只出结构、不判等级**；认不出的章节原样保留，一条不丢 |
| 否定 / 背景识别 Tool | `resume-negation` | 把「写了但没做过 / 只是背景」的条目挑出来，并给出为什么不算 | 纯规则；报告里原文照抄判定规则，可直接逐条核对 |
| 证据等级判定 Tool | `resume-evidence` | 给每条事实定级：有结果 / 有动作 / 仅提及，并汇总严宽两个口径 | 纯规则；`relaxed` 只影响「仅提及」算不算证据 |

四个工具的命令行与前端共用同一份实现，`--resume-agent` 走的也是同一份，不存在两套口径。

### 四种输入分别会做什么

| 输入 | 会联网吗 | 会调模型吗 | 说明 |
|------|----------|------------|------|
| `--cv-text` | 否 | 否 | 整段就是一个网址时自动改走网址分支 |
| `--cv-file` | 否 | 否 | 后缀决定怎么解析；文件不存在 / 后缀不认识 → 退出码 2 并说明原因 |
| `--cv-image` | 是（1 次模型调用 / 张） | 是（Qwen-VL） | 逐字转录，看不清的字写 `[看不清]`；没配 key 就跳过这一路 |
| `--cv-url` | 是（1 次 HTTP GET / 个） | 否 | 只抓你给的那个页面 |

### 两道阈值（写死在 `jd_agent/agents/resume_graph.py`，可直接检查）

| 常量 | 值 | 触发什么 |
|------|----|----------|
| `MIN_SOLID_FACTS` | 1 | 严格口径下一条硬证据都没有 → `Adjust`：换放宽口径重判一次（两个口径都不改字，只决定算不算证据） |
| `MIN_FACTS` | 3 | **可当证据**（没被挡掉）的条目少于 3 条 → `Ask`：问一句，而不是硬出一份清单 |

退出码与主线一致：`0` 已写出 Markdown（`Stop`）｜`3` 需要你回答一个问题（`Ask`）｜`2` 输入有误。

---

## 可选：接大模型（默认关闭）
## Polish Agent：基于已核验事实 + JD 要求，生成建议写法 / 追问预演 / 润色

前两个 Agent 各管一头：JD Agent 出「岗位要什么」，Resume Agent 出「我确实做过什么」。
Polish Agent 把两头合起来，回答最后一个问题 —— **这份简历对着这份 JD，该怎么写、会被追问什么、句子怎么顺**。
三样产出都由大模型写，但**模型只在规则已经钉死的事实范围内组织语言**：

```bash
python main.py --polish-agent --jd-file input/jd/xx.md --cv-file input/profile/xx.md   # 两路都给
python main.py --polish-agent --jd-text "岗位职责：……" --cv-file input/profile/xx.md    # 文本直接读取（不联网）
python main.py --polish-agent --jd-file … --cv-file … --polish-trace                    # 附带完整 Trace
python main.py --polish-agent --jd-file … --cv-file … --polish-out output/my_polish.md   # 换输出路径
```

JD 与简历**必须各给一路**：少给一路直接退出码 2 —— 只有 JD 只能得到要求清单，只有简历只能得到事实清单，
没有可对照的东西，硬写只能变成泛泛而谈。

产出长这样（`output/polish.md`）：

```markdown
# 李小明 × 某公司 · AI 应用开发实习生 · 写作简报与写法建议

> 由 Polish Agent（LangGraph）生成｜JD：input/jd/xx.md｜简历：input/profile/xx.md
> ｜规则先跑，模型只做表达：事实、等级与出处一字未改

**一句话结论**：12 条要求里 7 条对上了可当证据的事实（12 条能靠简历证明的要求中覆盖 58%）：5 条没对上证据，2 条正好写着「没做过」；已核验事实 4 条（硬证据 2 / 弱证据 2）。

## 写作简报（规则）

### 要求 ↔ 已核验事实（能力词典匹配）

| # | JD 要求 | 层级 | 最高证据 | 对上几条 | 缺口维度 | JD 出处 |
| --- | --- | --- | --- | --- | --- | --- |
| 9 | Docker 与部署 | 任职要求 | 无证据 | 0 | Docker、部署上线 | jd.md::L12 |

### 已核验事实（能当证据的）

### 不能当证据的内容（不许写成「做过」）

### 风险（模型看到的也是这一份）

- [事实冲突] JD 要「Docker 与部署」，简历里写的却是「了解向量数据库，没做过 Docker 部署」：这条别写，先补一段真实做过的经历

## 建议写法 [LLM]

## 追问预演 [LLM]

## 润色 [LLM]

## 跑批说明
```

### 图怎么走（`jd_agent/agents/polish_graph.py`）

两路来源分别排队（JD 那路先走完，再走简历那路），两路都取完了才进结构化；节点名就是 Trace 里的 `Action`：

| 节点 | 干什么 | 这一步的 Decision |
|------|--------|-------------------|
| `detect` | 识别两路输入类型，读不了的来源列出来并说明原因 | `Continue` |
| `read_text` / `parse_file` / `ocr_image` / `fetch_url` | 四种来源各自取证（文本 / 文件纯规则，图片走 Qwen-VL，网址抓一次） | `Continue` |
| `structure_jd` | 交给 **JD 结构化 Tool**：JD 原文 → 统一模板；要求由能力词典从原文匹配（带行号） | `Continue` |
| `structure_cv` | 交给 **简历结构化 Tool**：简历原文 → 章节 + 事实条目（带行号） | `Continue` |
| `screen` | 交给 **否定 / 背景识别 Tool**：先把「写了但没做过 / 只是背景」挡出来 | `Continue` |
| `evidence` | 交给 **证据等级判定 Tool**：剩下的条目才判「有结果 / 有动作 / 仅提及」 | `Continue` |
| `relax` | 严格口径下一条硬证据都没有：换**放宽口径**重判（不再读文件、不再发请求） | `Adjust` |
| `cap_match` | 交给 **能力词典匹配 Tool**：要求 ↔ 事实的对照 + 缺口 + 反向证据 + 风险排序 | `Continue` |
| `suggest` → `interview` → `polish` | 三块 LLM 内容，一块一步（各自记账、各自降级） | `Continue` |
| `finalize` | 写出 Markdown 与 Trace，写明 `StopReason` | `Stop` |
| `ask` | 要求认不出 / 一条能当证据的事实都没有 / 事实太少：问**一个**问题，一次模型都不调 | `Ask` |

### 分工：工具负责「模型能看到什么」，模型只负责「怎么说」

这是这一层最要紧的一条。模型**读不到 JD 全文，也读不到简历全文**，它只读一份由规则生成的写作简报 JSON：

| 简报里的字段 | 谁给的 | 模型拿它做什么 |
|--------------|--------|----------------|
| `requirements`（带 `jd_ref` 与 `evidence_level`） | 能力词典匹配 Tool | 决定**写什么方向** |
| `verified_facts`（带 `ref` 与 `level`） | 前面四个工具层层筛过 | 决定**能用哪些句子与数字** |
| `blocked_facts` | 否定 / 背景识别 Tool | **只许知道「这条不能写」** —— 不许写成「做过」 |
| `gaps` / `risks` | 能力词典匹配 Tool | 追问预演与风险提示对着它们来 |

换句话说：**「写了但没做过」的内容全在 `blocked_facts` 里，只告诉模型「别写」，不告诉它可以拿它当材料**。

### 五个工具（四个沿用 + 一个新增）

| 工具 | slug | 干什么 | 边界 |
|------|------|--------|------|
| 文件解析 Tool（复用） | `jd-file` | 本地 JD / 简历文件（md / txt / html / json / yaml）→ Markdown | 纯规则：**不联网、不读 `.env`、不调模型** |
| 简历结构化 Tool | `resume-struct` | 简历原文 → 章节 + 事实条目（带行号） | 纯规则 |
| 否定 / 背景识别 Tool | `resume-negation` | 把「写了但没做过 / 只是背景」的条目挑出来 | 纯规则；**不放宽口径** |
| 证据等级判定 Tool | `resume-evidence` | 逐条定级：有结果 / 有动作 / 仅提及 | 纯规则；`relaxed` 只影响「仅提及」算不算证据 |
| 能力词典匹配 Tool（新增） | `cap-match` | 要求 ↔ 事实的对照、缺口、反向证据；给两份材料再出一张对照表 | 纯规则；词表就是 `jd_agent/domain/lexicon.py`，**不新增词表、不做语义猜测** |

### 三块 LLM 内容（都会标 `[LLM]`）

| 块 | source | 只做这一件事 | 越界怎么办 |
|----|--------|--------------|------------|
| 建议写法 | `llm-suggest` | 用已核验事实拼一版可编辑的写法（3~5 句） | 出现「值得写 / 不值得写」这类结论词 → 整块丢弃 |
| 追问预演 | `llm-interview` | 针对要求、缺口与反向证据预演最可能被追问的 3 个问题 | 同上，整块丢弃 |
| 润色 | `llm-polish` | 一句原文对一句润色，信息不增不减 | 出现结论词 → 整块丢弃 |

三块彼此独立：某一块失败 / 超上限 / 没配 key，只把那一块标成「调用失败 / 已达调用上限 / 未启用」，
**不改 `StopReason`、不改退出码**，写作简报（规则部分）照常写出。

### 两道阈值（写死在 `jd_agent/agents/polish_graph.py` 与 `polish_brief.py`，可直接检查）

| 常量 | 值 | 触发什么 |
|------|----|----------|
| `MIN_SOLID_FACTS` | 1 | 严格口径下一条硬证据都没有 → `Adjust`：换放宽口径重判一次（否定与背景照旧不放宽） |
| `MIN_FACTS` | 3 | 可当证据的事实少于 3 条（或 JD 认不出要求）→ `Ask`：问一句，不硬写 |

退出码：`0` 已写出 Markdown（`Stop`）｜`3` 需要你回答一个问题（`Ask`）｜`2` 输入有误（少给一路来源 / 一路都读不出来）。

---


判断权始终在规则手里。大模型只在**规则报告已经生成之后**才介入，做三件规则做不好的事，
而且都要显式加开关才会联网：

| 层 | 开关 | 模型 | 只做这一件事 | 结果放在哪 |
|----|------|------|--------------|------------|
| 生成层 | `--llm` | `deepseek-chat` | ① 建议写法草稿（`source: llm-suggest`）② 报告润色（`llm-polish`）；③ 面试追问预演（`llm-interview`）**当前已闲置**（对效果不满意，待改版；把 `jd_agent/agents/generate.py` 的 `INTERVIEW_ENABLED` 改回 `True` 即恢复） | `Verdict.llm_report` 里的三个块，逐块标 `[LLM]`，**不替换**规则写的结论 / 理由 / 风险 / 建议写法 |
| 多模态层 | `--vision` | `qwen-vl-max` | 把项目描述里**引用的本地图片**读成文字事实 | 新增 `ProjectFact`（`source_file` 形如 `图片:assets/xxx.png`），**证据等级仍由 `classify()` 判定**，模型自述的等级一律丢弃 |

### 三条红线（都有对应测试，可直接逐条检查）

1. **规则优先**：词库能匹配的绝不调 LLM；v2.0 不做语义兜底（词典未命中也不调 LLM）；
   LLM 只在「规则报告已生成」之后介入，而且只读规则产出的结论 JSON，不读 JD / 项目原文；
2. **来源可区分**：每条 LLM 输出都带 `source`（`llm-suggest` / `llm-interview` / `llm-polish`），
   与 `source: rule` 严格区分；报告里每段都标 `[规则]` 或 `[LLM]`；
   已闲置的块不会被调用，但仍留在报告与 Trace 里，状态写「已闲置」；
3. **失败可降级**：LLM 挂了主流程照跑；不带 `--llm` 时行为与 v1.0**完全一致**；
   带 `--llm` 但调用失败 → 该块显示「未启用 / 调用失败」，主报告不受影响。

### 调用纪律（常量，可直接检查）

| 项 | 值 | 在哪 |
|----|----|------|
| 单次调用超时 | 30 秒 | `jd_agent/core/llm.py` 的 `DEFAULT_CALL_TIMEOUT` |
| 失败重试 | 1 次 | `DEFAULT_RETRIES` |
| 单次运行调用上限 | 10 次（含重试） | `DEFAULT_CALL_LIMIT`，可用 `--llm-max-calls` 调小 |
| 越界即丢 | 输出里出现「值得写 / 不建议写」这类档位词，整块丢弃并标「已丢弃」 | `jd_agent/agents/generate.py` 的 `VERDICT_WORDS` |

**批量时多个岗位共用同一份账本**：一次运行的调用总数仍然不超过 `--llm-max-calls`，
预算用完以后，后面岗位的 LLM 块会标成「已达调用上限」（不是「调用失败」），规则报告照旧。

LLM 不做结论、不改词典、不改 Decision：`Decision` 仍然只有 `Continue / Adjust / Ask / Stop`，
结论档位与 `StopReason` 只由规则给出。

### 配置 Key

在项目根目录新建 `.env`（已被 `.gitignore` 忽略，可参照 `.env.example`）：

```dotenv
DEEPSEEK_API_KEY=sk-你的-deepseek-key
DASHSCOPE_API_KEY=sk-你的-dashscope-key     # 也可以写 QWEN_API_KEY
```

可选覆盖项，不写就用默认值：

| 键 | 默认值 |
|----|--------|
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com/v1` |
| `DEEPSEEK_MODEL` | `deepseek-chat` |
| `DASHSCOPE_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `QWEN_VL_MODEL` | `qwen-vl-max` |
| `LLM_TIMEOUT` | `60`（秒）；生成层的 30 秒超时写死在 `DEFAULT_CALL_TIMEOUT`，不受它影响 |

`.env` 由 python-dotenv 的 `load_dotenv()` 加载，只补缺失的键。两条纪律：**已经存在的环境变量优先**，
`.env` 不会覆盖它们；终端日志只打印**键名**，从不回显值。

### 怎么用

```bash
python main.py --check-llm          # 先自检 key / 网络 / 模型名（加了 --vision 就一起检查多模态层）
python main.py --llm                # 规则结论 + 两块 LLM 内容（写法草稿 / 润色）
python main.py --llm --llm-max-calls 3          # 限制本次运行的 LLM 调用次数（默认 10，含重试）
python main.py --vision             # 额外解析项目描述里引用的图片
python main.py --llm --vision       # 两个都开
python main.py --vision --image D:\shots\dashboard.png          # 再补一张描述里没引用的图
python main.py --llm --text-model deepseek-reasoner --vision-model qwen-vl-plus   # 换模型
python main.py --llm --env D:\secrets\my.env                    # 换 .env 的位置
```

`--check-llm` 给每个启用的层发一条最小请求：通过返回 `0`，失败返回 `2` 并原样打印接口的报错。

### 硬边界（都会体现在 Trace 里）

- **默认完全离线**：不加 `--llm / --vision`，就不读 `.env`、不创建客户端、不发请求，输出与纯规则版本一致；
- **没有 Key 也不联网**：开了开关但缺 Key，三块都标「未启用」并记下原因，规则链路照走；
- **调用失败只降级**：网络 / 额度 / 模型名出错只写进该块的 `error` 与状态「调用失败」，
  **不改结论、不影响退出码**；
- **DeepSeek 只在报告生成之后调用**：Trace 里固定在 `Judge` 之后，依次
  `GenerateDraft -> RehearseInterview -> PolishReport`（追问预演已闲置，只保留 Trace 位置不发请求），
  这几步都是 `Continue`（还有活要干），
  最后由 `Finish` 落一条 `Stop`，把规则算出的 `StopReason` 原样带出来；
  不带 `--llm` 时这三步不存在，`Judge` 自己就是 `Stop`；
- **图片解析结果需人工复核**：`--vision` 会多一条 Trace（`EnrichWithVision`，位于 `ReadProject` 之后、
  `ExtractRequirements` 之前，同样有 Action / Observation / State Update / Decision 四要素），
  事实进的是同一个证据池；
- **依赖只有一个**：`.env` 交给 python-dotenv 读；两路 HTTPS 调用仍用标准库 `urllib`，
  不需要 openai / requests 这类 SDK。

---
## 批量：一次跑完 `input/jd` 里的所有岗位

不给 `--jd`（也就是直接 `python main.py`）时：

- `input/jd` 下**每个 `.md`、每个岗位**都跑一遍，一个岗位一份报告；
- 输出名带上岗位 id，例如 `output/resume_decision_3份校招_秋招_日常实习_AI岗位描述_1.md｜json｜html`；
  只有**一个**岗位时仍沿用 v1.0 的 `output/resume_decision.md｜json｜html`；
- 终端按岗位打印完整 Trace，最后给一张汇总表：

```text
批量结果汇总（4 个岗位，读自 2 个 JD 文件）
  [1] 岗位一：AI算法实习生（日常实习/秋招可投）｜核心覆盖 9/19｜值得写，但必须先改写｜Decision=Stop
  [3] 岗位三：AI产品实习生（校招/实习专属）｜核心覆盖 4/12｜暂不建议写｜Decision=Stop
```

- 识别不到任何岗位要求的文件会被跳过，并在终端里单独列出来；
- 只要有一个岗位是 `Ask`，整次运行就返回退出码 `3`；
- 加了 `--llm` 时**所有岗位共用同一份调用预算**（`--llm-max-calls`），预算用完的块标「已达调用上限」。

只想跑一个文件时照旧用 `--jd`：

```bash
python main.py --jd input/jd/后端开发工程师-岗位描述.md
```

---

## 工具区（`jd_agent/tools/`）

工具是「与 Agent 主链路解耦」的小工具：命令行（`jd_agent/cli.py`）与网页前端（`streamlit_app.py`）
都只从这一层取，不各自再写一套。`python main.py --help` 末尾会把注册表里的工具列出来。

| 文件 | 管什么 |
|------|--------|
| `tools/base.py` | `Tool` 基类（`slug / title / summary / usage` + `run()`）与 `ToolResult`（`ok / error / summary / notes / written`） |
| `tools/resume.py` | 简历排版：`ResumeTool` + 解析 / 渲染的纯函数 |
| `tools/resume_styles.py` | 三套排版风格（CSS + 两条 Markdown 约定） |
| `tools/jd_files.py` | 文件解析 Tool（`jd-file`）：本地 md / txt / html / json / yaml → Markdown 草稿 |
| `tools/jd_struct.py` | JD 结构化 Tool（`jd-struct`）：JD 原文 → 统一模板 |
| `tools/resume_struct.py` | 简历结构化 Tool（`resume-struct`）：简历原文 → 章节 + 抬头信息 + 事实条目 |
| `tools/resume_negation.py` | 否定 / 背景识别 Tool（`resume-negation`）：挑出「写了但没做过 / 只是背景」的条目 |
| `tools/resume_evidence.py` | 证据等级判定 Tool（`resume-evidence`）：有结果 / 有动作 / 仅提及 |
| `tools/cap_match.py` | 能力词典匹配 Tool（`cap-match`）：材料 → 命中的能力项；两份材料再加一张对照表 |
| `tools/__init__.py` | 注册表：`TOOLS` / `TOOL_BY_SLUG` / `get_tool(slug)` / `tool_help_lines()` |

三条约定：**一个工具只做一件事**；**输入有问题不抛异常**，用 `ToolResult(ok=False, error=…)` 说清楚；
**工具不读 `.env`、不发网络请求** —— 需要联网的能力属于 Agent 主链路，不属于工具区。

加一个新工具只要三步：继承 `Tool`、实现 `run()` 返回 `ToolResult`、把实例追加进 `TOOLS`。

---

## 简历排版（`--build-resume`）

把手写的简历 md 排成**简约大方**的两份文件：`output/resume.md` 与 `output/resume.html`
（内嵌 CSS、A4 宽度、无外链，浏览器里可直接打印成 PDF）。**纯规则**：不联网、不读 `.env`、不调用大模型。

```bash
python main.py --build-resume                            # 默认取 input/profile 下第一个非「示例」文件
python main.py --build-resume --resume-file input/profile/个人简历1.md
python main.py --build-resume --resume-style structure    # 换一套风格 → output/resume-structure.md / .html
python main.py --build-resume --resume-format html        # 只要 html（md,html 可用逗号组合）
python main.py --build-resume --resume-name 我的简历       # 改文件名 → output/我的简历.md / .html
python main.py --build-resume --resume-project input/project/示例项目1-AI周报助手.md   # 把项目描述并进「项目经历」
```

**三套风格**（`--resume-style`，默认 `classic`；Streamlit 页面上也能切）：

| key | 名字 | 差异 | 适合 |
|-----|------|------|------|
| `classic` | 经典简约 | 细灰线 + 无衬线，章节标题下一条细线 | 通用；粘到投递平台最稳（**默认**） |
| `structure` | 架构清晰 | 层级分明、模块分区，章节标题带色条，奖项荣誉双列展示 | 想让结构和层级一眼看清（只换 CSS，Markdown 不变） |
| `accent` | 强调竖线 | 章节标题带色条、表格去掉竖格线、章节之间加 `---`、子条目加粗 | 想多一点设计感 |

文件名默认跟着风格走：`classic → resume.md｜html`，其余 → `resume-structure.md｜html` / `resume-accent.md｜html`，
免得来回换风格互相覆盖；`--resume-name` 一旦显式给了就以它为准。

排版规则都写在 `jd_agent/tools/resume.py` 里，可以直接检查：

| 规则 | 说明 |
|------|------|
| 章节顺序 | 按「教育背景 → 项目经历 → 实习经历 → 专业技能 → 技能清单 → 奖项荣誉 → 竞赛与获奖 → 校园经历 → 基本信息 → 自我评价」重排；没写的章节不出现，模板外的章节按原文顺序排在后面，`自我评价` 这类总结性章节永远垫底 |
| 列表与表格 | 列表统一成 `- `；表格按最大列数补齐、分隔行统一成 `| --- |`、单元格里的 `|` 换成全角 |
| 空白 | 段落之间只留一个空行，行尾空白与重复空格清掉 |
| 不越界 | **只重排结构、规整空白**：不新增、不改写任何事实，也不判断「该不该写」；每次规整都会写进终端摘要与 HTML 末尾的「排版规整」 |

---

## Streamlit 前端（可选）

不想看终端就用网页版；它和命令行同源（同一套 `jd_agent`，同一套 `render.py` 产物）：

```bash
.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

- **分析结果**页：侧边栏选 JD（默认 `input/jd` 全部岗位，也可指定某个文件 / 某个岗位）、选项目描述或简历 md，
  可开关 `--llm` / `--vision`、限制调用次数；主区展示结论卡片、理由与风险、`[规则]` / `[LLM]` 分段、
  要求对照表，以及逐步展开的 Trace 四要素，并可下载 md / json / html；
- **Ask 直接在页面上答**：资料不够的岗位会就地给出问题和输入框，提交后把回答并入证据池、重新检索并判定
  （和命令行的 `--answer` 完全等价，Trace 里多一条 `ApplyUserAnswer`）；多岗位时只重跑你答的那个岗位，
  其余岗位保持原样。回答后这一轮不会自动联网；
- **简历排版**页：选简历 md（可多选项目描述并进「项目经历」）、选排版风格（classic / structure / accent），
  实时预览排好版的 HTML，并下载 md / html；风格与命令行同一套（`jd_agent/tools/resume_styles.py`），
  换风格只重排、不改内容，下载文件名也按风格取名；
- 配色在 `.streamlit/config.toml` 里配（浅色、克制），不用额外 CSS；
- 前端**不往磁盘写任何文件**：要落盘请用命令行。

---
## 输出怎么读

| 文件 | 用途 |
|------|------|
| `output/resume_decision.md` | 人看的报告：结论 / 理由 / 风险 / 建议写法 / 要求对照表 / Trace / 口径 |
| `output/resume_decision.html` | 同样内容的网页版，浏览器直接打开 |
| `output/resume_decision.json` | 结构化结果：每条匹配都带 `jd_ref`、`project_evidence[].ref`、`level`，方便二次加工；结论部分是 `verdict.source = rule`，用 `--llm` 时另有 `llm.blocks[]`（每块带 `source` / `status` / `calls`），用 `--vision` 时有 `vision` |
| `output/resume_decision_*.md｜json｜html` | 批量模式：多个岗位时按岗位分文件，后缀是 `jd_id`（如 `resume_decision_3份校招_秋招_日常实习_AI岗位描述_1.md`） |
| `output/resume.md｜resume.html` | `--build-resume` 默认（classic）排好版的简历：md 方便再编辑或粘贴，html 可直接打印 |
| `output/resume-<风格>.md｜html` | 换了风格时的默认文件名（如 `resume-structure.md`、`resume-accent.html`） |
| `output/jd.md` | `--jd-agent` 转出来的 JD（`--jd-out` 可改路径）：可直接当 `--jd` 的输入 |
| `output/jd.trace.md` | 加了 `--jd-trace` 时的完整 Trace（四要素 + StopReason） |
| `output/resume_facts.md` | `--resume-agent` 转出来的简历事实清单（`--cv-out` 可改路径）：抬头信息 + 一句话结论 + 按章节的证据表 + 被挡掉的条目 |
| `output/resume_facts.trace.md` | 加了 `--cv-trace` 时的完整 Trace |

Trace 同时会打印在终端上，并且**每个字段都能回查原文**（`文件::L行号`）。

---

## 目录结构

```text
AI_Job_Research_Agent/
├── main.py                  # 入口（PyCharm 直接 Run）
├── streamlit_app.py         # 可选：Streamlit 前端（复用同一套 jd_agent）
├── .streamlit/config.toml   # 可选：前端主题（浅色、克制）
├── requirements.txt         # 依赖：python-dotenv（可选大模型层）+ langgraph（三张 Agent 图）
├── .env.example             # 可选：接大模型时的 key 模板（.env 已被忽略）
├── input/
│   ├── jd/                  # 岗位描述（一份 JD；文件里可含多个岗位）
│   ├── project/             # 项目描述（一个项目一份文件）
│   └── profile/             # 简历 md：--build-resume 的输入，也能直接当 --project 的输入
├── output/                  # 报告输出
├── tests/
│   ├── test_agent.py        # 规则链路用例
│   ├── test_llm.py          # 可选大模型层用例（三红线 + 调用纪律 + 视觉层，注入假 client，不联网）
│   ├── test_batch_and_resume.py   # 批量岗位 + 简历排版用例
│   ├── test_jd_agent.py     # JD Agent 用例（注入假 opener / 假 client，不联网）
│   ├── test_resume_agent.py # Resume Agent 用例（事实抽取 / 证据等级 / 否定识别 / Agent Loop）
│   └── test_polish_agent.py  # Polish Agent 用例（能力词典匹配 / 写作简报 / 三块 LLM 内容 / Agent Loop）
└── jd_agent/
    ├── schema.py            # 数据结构（需求 / 事实 / 匹配 / Trace / 判断 / LLM 生成块）
    ├── lexicon.py           # 能力词典：自然语言 → 可对比的能力项
    ├── jd.py                # 读 JD → 岗位需要的能力（带行号）；一个文件里的多个岗位也拆得开
    ├── project.py           # 读项目描述 → 事实条目 + 证据等级
    ├── evidence.py          # 需求 ↔ 证据匹配（严格 / 放宽两种口径）
    ├── agent.py             # Agent Loop 与判定策略（所有阈值都在文件开头）
    ├── settings.py          # 可选：load_dotenv() 读 key 与覆盖项（只记键名，不回显值）
    ├── llm.py               # 可选：OpenAI 兼容接口客户端（标准库 urllib）
    ├── vision.py            # 可选：图片 → 文字事实（等级仍由规则判定）
    ├── generate.py          # 可选：报告生成后的 LLM 内容块（草稿 / 追问预演（已闲置）/ 润色）
    ├── jd_html.py           # HTML → Markdown（标准库解析，无第三方依赖）
    ├── jd_source.py         # 四种来源（文本 / 文件 / 图片 / 网址）的识别与取证
    ├── jd_graph.py          # JD Agent（LangGraph）：JD → 统一模板 Markdown
    ├── resume_facts.py      # 简历 → 事实条目；否定 / 背景 / 课程 / 证据等级的规则层
    ├── resume_graph.py      # Resume Agent（LangGraph）：简历 → 事实清单 + 证据等级
    ├── polish_brief.py      # Polish Agent 的规则层：要求 ↔ 事实对照 + 缺口 + 反向证据 + 风险
    ├── polish_llm.py        # 三块 LLM 内容（建议写法 / 追问预演 / 润色）的提示词、校验与降级
    ├── polish_graph.py      # Polish Agent（LangGraph）：已核验事实 + JD 要求 → 写法建议
    ├── tools/               # 工具区：命令行与前端共用，一个工具 = Tool 子类 + run()
    │   ├── base.py          # Tool / ToolResult：工具的公共契约
    │   ├── resume.py        # 简历排版：读简历 md → 有序的 md / html（纯规则）+ ResumeTool
    │   ├── resume_styles.py # 三套排版风格（CSS + Markdown 约定）
    │   ├── jd_files.py      # 文件解析 Tool（jd-file）
    │   ├── jd_struct.py     # JD 结构化 Tool（jd-struct）
    │   ├── resume_struct.py     # 简历结构化 Tool（resume-struct）
    │   ├── resume_negation.py   # 否定 / 背景识别 Tool（resume-negation）
    │   ├── resume_evidence.py   # 证据等级判定 Tool（resume-evidence）
    │   ├── cap_match.py         # 能力词典匹配 Tool（cap-match）
    │   └── __init__.py      # 注册表：TOOLS / get_tool / tool_help_lines
    ├── render.py            # md / json / html / 终端渲染
    └── cli.py               # 命令行入口（批量 / --build-resume / 三个 Agent 模式 / 单次分析）
```

---

## 测试

```bash
python -m unittest discover -s tests -v
```

覆盖：JD 切分与层级去重、项目事实定级（含背景降级与否定识别）、严格 / 放宽两种匹配、子维度覆盖、
Trace 四要素完整性、Ask / Answer / Stop 三条路径、CLI 退出码与三种输出格式、批量岗位与简历排版。

大模型层的用例全程不联网（注入假 client，一个请求都不发），覆盖：

- 默认关闭时 Trace、结论与纯规则版本逐字一致，且不会创建任何客户端；
- 三块内容的 `source` 与所在位置（固定在 `Judge` 之后）、`[规则]` / `[LLM]` 段落标记；
- 「LLM 不做结论」：输出里出现档位词时整块丢弃；
- 失败可降级：调用失败重试 1 次后标「调用失败」、缺 Key 标「未启用」、超上限停手，
  三种情况下规则结论与退出码都不变；
- 调用纪律：单次超时 30 秒、上限 10 次、`Ask` 路径一次都不调模型；
- 视觉层：模型输出 → 规则定级、图片引用发现、失败只留 note；
- 批量：一个文件里的多个岗位各出一份报告（文件名带 `jd_id`）、`--jd` 指定单文件时仍是 v1.0 行为、
  多个岗位共用一份 LLM 预算、只要有 `Ask` 就返回退出码 `3`、没有要求的文件被跳过；
- 简历排版：章节重排与内容保真、表格补列、HTML 自包含且样式全部限定在 `.resume-doc` 里、
  项目描述并入「项目经历」、`--build-resume` 全程不创建客户端；
- 工具区与三套风格：`RESUME_TOOL` 注册在 `TOOLS` 里、`ToolResult` 可序列化、`run()` 遇到坏输入
  （文件缺失 / 格式不认识 / 风格不认识）返回 `ok=False` 而不抛异常；三套风格的 `<style>` 互不相同、
  只有 `accent` 动 Markdown（章节之间加 `---`、子条目改加粗行）、默认 `classic` 的 md 与 html
  保持稳定的通用排版（回归锁）。

JD Agent 的用例同样全程不联网（网址注入假 opener、图片注入假 client）：

- 三种输入的识别与分支：粘贴的网址自动改走网址分支、后缀决定文件怎么解析、没配 key 的图片直接跳过；
- HTML → Markdown 的结构保真（标题 / 列表 / 表格）、script 与样式丢弃、严格口径丢 UI 噪声而放宽口径全留；
- 来源失败只降级：一个来源读不到不影响其它来源，最后如实写进 `StopReason`；
- 图片转录失败 → 降级后 `Ask`；页面是前端渲染的空壳 → `Ask`；要点少于 3 条 → `Ask`，都不硬编内容；
- 网址分支的 `Adjust` 只重抽一次、**不重复抓网络**；每一步四要素齐全、Decision 只在四种里、图里的节点与文档一致；
- 转出来的 JD Markdown 能被主线解析（岗位 / 公司 / 能力都对得上），且再排一次结果不变（幂等）；
- 命令行：`--jd-out` / `--jd-trace` / `--quiet` 与退出码 0 / 3 / 2（没有来源、来源不可用都算 2）；
- 工具区：两个新工具注册在 `TOOLS` 里，输入有问题返回 `ok=False` 而不抛异常。

Resume Agent 的用例同样全程不联网（网址注入假 opener、图片注入假 client）：

- 事实抽取：抬头信息与「求职意向」重复时后写胜、章节角色与章节顺序、行号与原文对得上、表格 / 有序列表也读得进来；
- 否定 / 背景：关键词前的否定（「未接触过向量数据库」）与写在后面的否定（「了解 TensorFlow 但没做过完整项目」）都认，
  逗号后面的否定不连坐前面（「熟悉 Python，没写过 C++」）、「经历：无」这类缺失写法也认、整节背景会屏蔽节内条目、
  「教育背景」「个人简介」不会被当成背景整节作废；
- 等级判定：荣誉与名次算「有结果」、课程 / 培训且没写指标时封顶「仅提及」（写了指标就不封）、
  三个等级各有一个例子钉住；
- 顺序与口径：`ScreenFacts` 一定在 `JudgeEvidence` 之前；`Adjust` 只重判一次且**放宽也不会把否定 / 背景放进来**；
  纯「了解 / 熟悉」的简历会走一次 `Adjust` 后 `Stop`；一条能当证据的都没有时 `Ask`；
- 四种输入：文本与文件分支一次网络请求都不发、一次模型都不调；文件分支的出处写成真实文件路径（`cv.md::L行号`）；
  图片分支用简历版提示词调 Qwen-VL；一个来源读不到不影响其它来源；合并多来源时写明「行号按合并后的文本计」；
- 渲染：事实清单可重渲染（同样的输入 → 同样的输出）、Trace 四要素齐全且节点与文档一致；
- 命令行：`--cv-out` / `--cv-trace` / `--quiet` 与退出码 0 / 3 / 2，`--cv-file` 与排版用的 `--resume-file` 互不干扰；
- 工具区：三个新工具注册在 `TOOLS` 里（slug 不重复），输入有问题返回 `ok=False` 而不抛异常。

Polish Agent 的用例同样全程不联网（网址注入假 opener、图片注入假视客户端、三块内容注入假文本客户端）：

- 能力词典匹配：命中带行号与关键词、按能力项聚合去重、材料为空 / 没命中都给得出话；
- 要求 ↔ 事实对照：对上 / 缺口 / 反向证据分得开，`matched + gaps` 恰好等于能证明的要求数，覆盖率算得准；
- 风险顺序：反向证据 > 缺口 > 弱证据 > 只有动作没结果；反向证据不会再被报一遍「完全没提」（不自相矛盾）；
- 送进模型的 JSON：只有 `jd_ref` / `ref` / `level` 这些规则给的东西，**没有 JD / 简历全文**，被挡掉的条目也不会出现在 `verified_facts` 里；
- 三块内容的解析与降级：越界出结论词 → 整块丢弃；调用失败 → 「调用失败」；超上限 → 「已达调用上限」；
  缺 Key → 「未启用」且一次请求都不发（假客户端调用次数为 0）；三种情况都不改 `StopReason` 与退出码；
- 行动序列：两路文本正好走 `DetectSource → ReadText ×2 → StructureJD → StructureResume → ScreenFacts →
  JudgeEvidence → MatchCapabilities → SuggestWording → RehearseInterview → PolishWording → WriteSuggestions`；
  JD 那路先取、`ScreenFacts` 一定在 `JudgeEvidence` 之前；
- `Adjust` 只走一次且只放宽证据强弱：全是「没做过」的简历放宽后依然一条证据都没有，最后 `Ask`；
- `Ask` 四种情形：少给一路来源 / 认不出要求 / 一条能当证据的都没有 / 可当证据的事实少于 3 条 —— 都不调模型；
- 四种输入：文本与文件分支一次网络请求都不发、一次模型都不调；JD 截图用 JD 提示词、简历截图用简历提示词；
- 命令行：`--polish-out` / `--polish-trace` / `--quiet` 与退出码 0 / 3 / 2；这个模式一律读 `.env`（三块内容要文本层 key）；
- 工具区：`cap-match` 注册在 `TOOLS` 里（slug 不重复），坏输入返回 `ok=False` 而不抛异常。

`.env` 解析且不覆盖已有环境变量。

---

## 想改哪里

- **加一项能力**：在 `jd_agent/domain/lexicon.py` 的 `CAPABILITIES` 里加一条 `Capability`（`key / name / category / keywords / subs`），并给它一个**具体**的关键词——越泛的词（比如「整理」）越容易误命中；
- **调判定松紧**：`jd_agent/agents/agent.py` 开头的 `COVERAGE_OK / COVERAGE_MIN / MIN_FACTS / MAX_ROUNDS`；
- **调证据等级**：`jd_agent/domain/project.py` 里的动作词与交付词表，以及 `## 背景` 那类段落的降级规则。
- **调 LLM 做什么 / 调用纪律**：`jd_agent/agents/generate.py`（三块任务的提示词、`VERDICT_WORDS`）与
  `jd_agent/core/llm.py`（`DEFAULT_CALL_TIMEOUT` / `DEFAULT_RETRIES` / `DEFAULT_CALL_LIMIT`）。
- **调简历排版**：`jd_agent/tools/resume.py` 的 `SECTION_ORDER`（章节顺序）；
- **加一套排版风格**：在 `jd_agent/tools/resume_styles.py` 的 `STYLES` 里加一条 `ResumeStyle`
  （`key / name / summary / css`，还有章节分隔线与子条目写法两个开关），命令行与前端会自动多一个选项；
- **加一个工具**：继承 `jd_agent/tools/base.py` 的 `Tool`、实现 `run()` 返回 `ToolResult`，
  再把实例追加进 `jd_agent/tools/__init__.py` 的 `TOOLS`；
- **调 JD Agent 的严宽**：`jd_agent/agents/jd_graph.py` 的 `MIN_URL_LINES` / `MIN_JD_BULLETS`；
- **调页面噪声词表**：`jd_agent/domain/jd_html.py` 的 `NOISE_WORDS`（严格口径只对短行生效，长句里出现同一批词不会被误伤）；
- **给 JD Agent 加一种输入类型**：在 `jd_agent/services/jd_source.py` 加一个 `SOURCE_*` 与 `detect_sources` 的一个分支，
  再到 `jd_graph.py` 的 `ROUTES` 里挂一个节点 —— 图的形状与 Trace 格式都不用改；
- **调 Resume Agent 的严宽**：`jd_agent/agents/resume_graph.py` 的 `MIN_FACTS`（可当证据的条目少于几条就 `Ask`）/
  `MIN_SOLID_FACTS`（硬证据少于几条就先 `Adjust` 一次）；
- **调「挡伪装」的识别范围**（都写在 `jd_agent/domain/resume_facts.py`，可直接逐条改）：
  `STRONG_NEGATION_RE`（「未接触 / 没做过」这类否定写法）、`ABSENCE_RE`（「经历：无」这类缺失写法）、
  `BACKGROUND_SENTENCE_RE` 与 `PLAN_RE`（「旨在…」「计划学习…」）、`COURSE_RE`（课程 / 培训口径）；
- **调简历章节怎么归档**：`jd_agent/domain/resume_facts.py` 的 `SECTION_ROLE`（章节名 → 头部 / 教育 / 经历 / 技能…）、
  `BACKGROUND_SECTION_RE`（哪些章节名整节按背景处理）与 `AWARD_MARKERS`（荣誉口径：默认算「有结果」）；
- **改前端**：`streamlit_app.py`（布局与交互）配合 `.streamlit/config.toml`（配色）。

---

## 局限（请务必知道）

- **词典匹配不懂语义**：「了解」「用过」「熟练」这类程度差异只能靠证据等级粗略区分，报告里的每条理由都带了原文行号，请人工复核；
- **它判断的是「能不能撑住这条要求」，不是「你水平多高」**：项目描述写得实，判断才准——这也是它先问你问题的原因；
- **规则阈值是拍出来的**：60% / 35% / 3 条这些数字是为了可解释和可调，不是行业标准；
- **一次只判断一个项目**：想比较几个项目，分别跑，然后比 `coverage` 与风险条数。
- **图片解析结果需人工复核**：`--vision` 让模型把图里的数字、指标读成文字事实，它可能看错刻度、漏读单位 ——
  证据等级虽然由规则判定，事实本身仍要对着原图核一遍再写进简历；
- **LLM 生成内容不是判定依据**：`--llm` 产出的写法草稿、追问预演与润色都由模型生成，
  只用来准备面试与改简历；「值不值得写」这个结论、`StopReason` 与每一步的 `Decision` 始终来自规则。
  v2.0 不做语义兜底：词典没命中的长尾能力不会被模型补进证据池，报告里也不会出现模型「猜」出来的证据。
- **简历排版只负责排版**：`--build-resume` 不替你改措辞、不补数字、不排序项目，只重排结构与空白；
  换风格也只是换排版，一个字的内容都不会变；内容写得实不实，仍然要你自己对着主线那份报告来判断。
- **网址抓取只是「单页快照」**：前端渲染（正文靠 JS 出）或需要登录的页面抓不到内容 —— 这时它会 `Ask` 让你补文本或截图，
  而不是假装读到了；放宽口径会把页面噪声一起留下，转出来的 JD 请过一眼；
- **图片转录要人工复核**：Qwen-VL 逐字转录出来的数字、专有名词与 `[看不清]` 处，投递前请对着原图核一遍；
- **JD Agent 只做「搬运 + 归位」**：不改写措辞、不补内容、不判断岗位要什么能力 —— 那是主线（judge）的事。
- **Resume Agent 只拆「简历里写了什么」**：不改措辞、不补数字、不替你把「参与」改写成「负责」；
  等级算的是原文写了什么，不是你实际水平多高；
- **否定 / 背景识别是词表驱动的**：它只认写在词表里的写法（`jd_agent/domain/resume_facts.py` 的 `STRONG_NEGATION_RE` /
  `ABSENCE_RE` / `BACKGROUND_SENTENCE_RE` / `PLAN_RE` / `COURSE_RE`），所以「没怎么做过」这类没进词表的说法可能漏挡；
  被挡掉的条目都带原文与行号，扫一眼「不能当证据的内容」那张表就能核对；反过来它也不会误放开：否定与背景永远不算证据；
- **它只挡「写了但没做过」，不帮你找「做过但没写」**：没写在简历里的经历它不会替你补，只能靠 `Ask` 问你一句。
- **Polish Agent 只负责「怎么说」，不负责「能不能写」**：三块内容都由模型生成，但它只能组织**规则已经核验过的**
  句子与数字；事实、等级与出处一字未改。投递前请对着「写作简报」与出处逐条核一遍 ——
  它不会替你凭空补一段经历，也不会把「没做过」写成「做过」（被挡掉的条目只告诉模型「别写」）；
- **建议写法的好坏取决于证据本身**：可当证据的事实少于 3 条，或 JD 里一条要求都认不出来，它会先 `Ask`，
  而不是硬凑一版听起来不错的写法；能力词典没命中的长尾要求也不会被模型补进来。
