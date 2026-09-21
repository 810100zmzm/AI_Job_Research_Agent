# AI 求职尽调 Agent · 这个项目值不值得写进简历？

做过几个 AI 项目，却说不清哪个真正适合写进简历去投 AI 工程师岗；更担心的是，面试官顺着简历往下问，自己答不上来。

这个 Agent 只做一件事：

> 读 **一份 JD** + **一个项目描述** → 抽出岗位真正需要的能力 → 到项目里逐条找能支撑这些能力的证据
> → 给出**要不要写进简历**的判断（2~3 条理由 + 1~2 个风险），并留下可以逐条检查的运行 Trace。

它不猜：资料不够时停下来问你**一个**最关键的问题；证据够了才下结论，并写明 `StopReason`。

**纯离线** —— 规则 + 能力词典（`jd_agent/lexicon.py`），不需要 API Key：不加开关就一个请求都不发，也不读 `.env`。
需要「建议写法草稿」和「图片里的证据」时再加 `--llm` / `--vision`（见下文「可选：接大模型」）。

v1.1 顺手补了两件事：**批量**（不指定 `--jd` 就跑完 `input/jd` 下所有文件的**全部岗位**，一个岗位一份报告）
和**简历排版**（`--build-resume` 把手写的简历排成简约大方的 md / html）；想要网页版还有 Streamlit 前端。

v2.0 把简历排版收进了**工具区 `jd_agent/tools/`**：每个工具都是一个 `Tool` 子类、统一返回 `ToolResult`，
命令行与前端共用同一份实现，不再各写一套；排版另加**三套风格**（`--resume-style classic|compact|accent`）。
默认那套（classic）的输出与 v1.1 **逐字一致** —— 不换风格就不用改任何习惯。

---

## 快速开始

### 在 PyCharm 里

1. `File > Open` → 选择项目目录 `D:\codexAbout\projects\AI_Job_Research_Agent`
2. 解释器选任意 Python 3.9+，然后 `pip install -r requirements.txt`（只有一个 python-dotenv）
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
python main.py --build-resume --resume-style compact   # 换风格（classic / compact / accent）→ resume-compact.*
python main.py --llm                               # 可选：报告生成后补写法草稿 / 润色（需 .env）
python main.py --vision                            # 可选：Qwen-VL 把图片读成文字事实（需 .env）

.venv\Scripts\python.exe -m streamlit run streamlit_app.py      # 可选：网页版前端
```

**要分析哪份 md**：JD 用 `--jd`（不给就是 `input/jd` 下全部岗位），第二个输入用 `--project`。
第二个输入不限于 `input/project/`，你自己的简历 md 也能直接喂进去，例如：

```bash
python main.py --project input/profile/资料不足项目测试简历1.md
```

**退出码**：`0` 已给出判断｜`3` 需要你回答一个问题｜`2` 输入有误。

---

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

---

## 可选：接大模型（默认关闭）

判断权始终在规则手里。大模型只在**规则报告已经生成之后**才介入，做三件规则做不好的事，
而且都要显式加开关才会联网：

| 层 | 开关 | 模型 | 只做这一件事 | 结果放在哪 |
|----|------|------|--------------|------------|
| 生成层 | `--llm` | `deepseek-chat` | ① 建议写法草稿（`source: llm-suggest`）② 报告润色（`llm-polish`）；③ 面试追问预演（`llm-interview`）**当前已闲置**（对效果不满意，待改版；把 `jd_agent/generate.py` 的 `INTERVIEW_ENABLED` 改回 `True` 即恢复） | `Verdict.llm_report` 里的三个块，逐块标 `[LLM]`，**不替换**规则写的结论 / 理由 / 风险 / 建议写法 |
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
| 单次调用超时 | 30 秒 | `jd_agent/llm.py` 的 `DEFAULT_CALL_TIMEOUT` |
| 失败重试 | 1 次 | `DEFAULT_RETRIES` |
| 单次运行调用上限 | 10 次（含重试） | `DEFAULT_CALL_LIMIT`，可用 `--llm-max-calls` 调小 |
| 越界即丢 | 输出里出现「值得写 / 不建议写」这类档位词，整块丢弃并标「已丢弃」 | `jd_agent/generate.py` 的 `VERDICT_WORDS` |

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
python main.py --build-resume --resume-style compact      # 换一套风格 → output/resume-compact.md / .html
python main.py --build-resume --resume-format html        # 只要 html（md,html 可用逗号组合）
python main.py --build-resume --resume-name 我的简历       # 改文件名 → output/我的简历.md / .html
python main.py --build-resume --resume-project input/project/示例项目1-AI周报助手.md   # 把项目描述并进「项目经历」
```

**三套风格**（`--resume-style`，默认 `classic`；Streamlit 页面上也能切）：

| key | 名字 | 差异 | 适合 |
|-----|------|------|------|
| `classic` | 经典简约 | 细灰线 + 无衬线，章节标题下一条细线 | 通用；粘到投递平台最稳（**默认，输出与 v1.1 逐字一致**） |
| `compact` | 紧凑一页 | 字号与行距更紧、章节标题带浅底块 | 内容多、想压到一页（只换 CSS，Markdown 不变） |
| `accent` | 强调竖线 | 章节标题带色条、表格去掉竖格线、章节之间加 `---`、子条目加粗 | 想多一点设计感 |

文件名默认跟着风格走：`classic → resume.md｜html`，其余 → `resume-compact.md｜html` / `resume-accent.md｜html`，
免得来回换风格互相覆盖；`--resume-name` 一旦显式给了就以它为准。

排版规则都写在 `jd_agent/tools/resume.py` 里，可以直接检查：

| 规则 | 说明 |
|------|------|
| 章节顺序 | 按「基本信息 → 教育背景 → 技能清单 → 项目经历 → 实习经历 → 荣誉奖项 → 校园经历 → 自我评价」重排；没写的章节不出现，模板外的章节按原文顺序排在后面，`自我评价` 这类总结性章节永远垫底 |
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
- **简历排版**页：选简历 md（可多选项目描述并进「项目经历」）、选排版风格（classic / compact / accent），
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
| `output/resume-<风格>.md｜html` | 换了风格时的默认文件名（如 `resume-compact.md`、`resume-accent.html`） |

Trace 同时会打印在终端上，并且**每个字段都能回查原文**（`文件::L行号`）。

---

## 目录结构

```text
AI_Job_Research_Agent/
├── main.py                  # 入口（PyCharm 直接 Run）
├── streamlit_app.py         # 可选：Streamlit 前端（复用同一套 jd_agent）
├── .streamlit/config.toml   # 可选：前端主题（浅色、克制）
├── requirements.txt         # 唯一依赖：python-dotenv
├── .env.example             # 可选：接大模型时的 key 模板（.env 已被忽略）
├── input/
│   ├── jd/                  # 岗位描述（一份 JD；文件里可含多个岗位）
│   ├── project/             # 项目描述（一个项目一份文件）
│   └── profile/             # 简历 md：--build-resume 的输入，也能直接当 --project 的输入
├── output/                  # 报告输出
├── tests/
│   ├── test_agent.py        # 规则链路用例
│   ├── test_llm.py          # 可选大模型层用例（三红线 + 调用纪律 + 视觉层，注入假 client，不联网）
│   └── test_batch_and_resume.py   # 批量岗位 + 简历排版用例
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
    ├── tools/               # 工具区：命令行与前端共用，一个工具 = Tool 子类 + run()
    │   ├── base.py          # Tool / ToolResult：工具的公共契约
    │   ├── resume.py        # 简历排版：读简历 md → 有序的 md / html（纯规则）+ ResumeTool
    │   ├── resume_styles.py # 三套排版风格（CSS + Markdown 约定）
    │   └── __init__.py      # 注册表：TOOLS / get_tool / tool_help_lines
    ├── render.py            # md / json / html / 终端渲染
    └── cli.py               # 命令行入口（批量 / --build-resume / 单次分析）
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
  正文与重构前**逐字一致**（回归锁）。

`.env` 解析且不覆盖已有环境变量。

---

## 想改哪里

- **加一项能力**：在 `jd_agent/lexicon.py` 的 `CAPABILITIES` 里加一条 `Capability`（`key / name / category / keywords / subs`），并给它一个**具体**的关键词——越泛的词（比如「整理」）越容易误命中；
- **调判定松紧**：`jd_agent/agent.py` 开头的 `COVERAGE_OK / COVERAGE_MIN / MIN_FACTS / MAX_ROUNDS`；
- **调证据等级**：`jd_agent/project.py` 里的动作词与交付词表，以及 `## 背景` 那类段落的降级规则。
- **调 LLM 做什么 / 调用纪律**：`jd_agent/generate.py`（三块任务的提示词、`VERDICT_WORDS`）与
  `jd_agent/llm.py`（`DEFAULT_CALL_TIMEOUT` / `DEFAULT_RETRIES` / `DEFAULT_CALL_LIMIT`）。
- **调简历排版**：`jd_agent/tools/resume.py` 的 `SECTION_ORDER`（章节顺序）；
- **加一套排版风格**：在 `jd_agent/tools/resume_styles.py` 的 `STYLES` 里加一条 `ResumeStyle`
  （`key / name / summary / css`，还有章节分隔线与子条目写法两个开关），命令行与前端会自动多一个选项；
- **加一个工具**：继承 `jd_agent/tools/base.py` 的 `Tool`、实现 `run()` 返回 `ToolResult`，
  再把实例追加进 `jd_agent/tools/__init__.py` 的 `TOOLS`；
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
