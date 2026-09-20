"""能力词典：把 JD 与项目描述里的自然语言，映射成可对比的能力项。

这一层是「离线可用、可检查」的关键：不依赖大模型也能稳定抽出「岗位需要什么能力」。
想扩展能力项时，直接往 CAPABILITIES 里加一条 Capability 即可。
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .schema import Capability, SubItem


def _sub(name: str, *keywords: str) -> SubItem:
    return SubItem(name=name, keywords=tuple(keywords))


CAPABILITIES: Tuple[Capability, ...] = (
    # ---------- 准入门槛 ----------
    Capability(
        key="education",
        name="学历与专业要求",
        category="准入门槛",
        keywords=("本科", "硕士", "博士", "在读", "应届", "专业", "学历"),
        subs=(
            _sub("学历层次", "本科", "硕士", "博士", "研究生", "学历"),
            _sub("专业匹配", "计算机", "人工智能", "软件工程", "数学", "统计", "电子信息", "经管", "专业"),
        ),
        actions=("若专业不完全对口，用「相关课程 + 项目 + 竞赛」三件事补齐证明，并在简历首行标注学历与专业",),
    ),
    Capability(
        key="cohort",
        name="届别与投递身份",
        category="准入门槛",
        keywords=("25届", "26届", "27届", "应届", "校招", "秋招", "在校生", "届别", "招聘对象"),
        actions=("在简历首行写明届别与可投身份（校招 / 日常实习 / 暑期实习）",),
    ),
    Capability(
        key="stability",
        name="实习时长与稳定性",
        category="准入门槛",
        keywords=(
            "稳定实习", "长期稳定", "实习时长", "实习时间", "实习期限", "可实习", "连续实习",
            "实习2个月", "实习三个月", "实习六个月", "出勤", "实习天数",
        ),
        actions=("明确写清可实习的起止时间与每周天数（例如「2026.03 起，每周 4 天，可连续 6 个月」）",),
    ),
    # ---------- 硬技能 ----------
    Capability(
        key="python",
        name="Python 编程能力",
        category="硬技能",
        keywords=("python",),
        actions=("保持手感：每周 2-3 道题 + 一个小脚本沉淀到 GitHub，简历用「用 Python 做过什么」而不是「会 Python」",),
    ),
    Capability(
        key="dl_framework",
        name="深度学习框架（PyTorch / TensorFlow）",
        category="硬技能",
        keywords=("pytorch", "tensorflow", "深度学习框架", "框架训练"),
        subs=(_sub("PyTorch", "pytorch"), _sub("TensorFlow", "tensorflow")),
        actions=("用 PyTorch 独立复现一个完整训练流程（数据加载 → 训练 → 评估 → 保存），把代码与实验记录开源",),
    ),
    Capability(
        key="ml_basics",
        name="机器学习 / 深度学习基础理论",
        category="算法基础",
        keywords=(
            "机器学习", "深度学习", "神经网络", "分类", "回归", "聚类", "算法基础", "模型原理",
        ),
        subs=(
            _sub("机器学习基础", "机器学习", "分类", "回归", "聚类", "神经网络"),
            _sub("深度学习", "深度学习", "神经网络", "cnn", "rnn", "transformer"),
        ),
        actions=("系统复习 ML 基础，产出一份可自测的知识卡片（面试高频：过拟合、评估指标、损失函数）",),
    ),
    Capability(
        key="nlp_cv",
        name="NLP / CV 方向基础任务",
        category="算法基础",
        keywords=("nlp", "自然语言处理", "cv", "计算机视觉", "图像", "文本", "语音", "多模态"),
        actions=("选一个方向跑通公开 baseline 并写复现笔记（数据 → 模型 → 指标 → 结论）",),
    ),
    # ---------- 大模型 ----------
    Capability(
        key="llm_basics",
        name="大模型基础原理与认知",
        category="大模型应用",
        keywords=("大模型", "llm", "生成式", "aigc", "transformer", "预训练", "智能问答", "智能产品"),
        actions=("整理一份大模型基础笔记（Transformer / 预训练 / 推理 / 幻觉），能在面试里讲清楚链路",),
    ),
    Capability(
        key="rag",
        name="RAG 与知识库应用",
        category="大模型应用",
        keywords=("rag", "知识库", "检索增强", "检索", "问答", "知识库录入"),
        subs=(
            _sub("知识库搭建", "知识库", "rag", "检索增强", "问答"),
            _sub("向量检索", "向量数据库", "向量库", "embedding", "faiss", "milvus", "chroma", "检索"),
        ),
        actions=("用 LangChain / LlamaIndex 搭一个最小可用 RAG（切分 → 向量化 → 召回 → 引用）并写 README",),
    ),
    Capability(
        key="finetune",
        name="模型微调",
        category="大模型应用",
        keywords=("微调", "finetune", "fine-tune", "lora", "sft", "指令微调"),
        actions=("用 LoRA 微调一个开源小模型，记录数据构造方式与前后效果对比",),
    ),
    Capability(
        key="prompt",
        name="Prompt 工程",
        category="大模型应用",
        keywords=("prompt", "提示词"),
        actions=("沉淀一份 Prompt 对照实验记录（bad case → 改法 → 指标变化）",),
    ),
    # ---------- 工程实践 ----------
    Capability(
        key="web_api",
        name="接口 / Web 后端开发",
        category="工程实践",
        keywords=("fastapi", "flask", "web开发", "接口开发", "接口联调", "封装基础模型接口", "后端", "服务端", "api"),
        actions=("用 FastAPI 把模型或 RAG 流程封装成 HTTP 接口（带流式输出更好）",),
    ),
    Capability(
        key="debug_eng",
        name="调试与问题排查",
        category="工程实践",
        keywords=("调试", "bug", "排查", "联调", "测试验证", "问题定位", "定位问题", "问题清单"),
        actions=("留一份排障记录：现象 → 日志 → 定位 → 修复 → 回归，面试可直接讲",),
    ),
    Capability(
        key="data_process",
        name="数据处理与清洗",
        category="工程实践",
        keywords=("数据处理", "数据清洗", "样本筛选", "数据集", "数据整理", "数据质量", "标注"),
        actions=("写一个可复用清洗脚本（去重 / 过滤 / 格式统一 / 输出统计报告）",),
    ),
    Capability(
        key="experiment",
        name="实验与效果评估复盘",
        category="工程实践",
        keywords=("效果评估", "实验", "复盘", "准确率", "调优", "参数调优", "实验日志", "结果对比", "指标"),
        actions=("建立实验记录模板（配置 / 指标 / 结论），把一次真实调优过程写成可展示的材料",),
    ),
    Capability(
        key="tools_git_linux",
        name="Linux / Git 工程工具",
        category="工程实践",
        keywords=("linux", "git", "版本管理", "命令行"),
        subs=(_sub("Linux", "linux", "命令行"), _sub("Git", "git", "版本管理")),
        actions=("完整走一遍 Git 协作流程（分支 / PR / 冲突解决），并熟悉 Linux 常用命令",),
    ),
    Capability(
        key="docker",
        name="Docker 与部署",
        category="工程实践",
        keywords=("docker", "容器", "部署", "上线"),
        subs=(_sub("Docker", "docker", "容器"), _sub("部署上线", "部署", "上线")),
        actions=("把个人项目打成镜像并在云服务器上跑通，README 附部署步骤",),
    ),
    Capability(
        key="frontend",
        name="前端 / 全栈能力",
        category="工程实践",
        keywords=("前端", "全栈", "页面", "小程序", "web页面", "可视化页面"),
        actions=("补基础前端：能改页面、能接接口、能部署静态站点",),
    ),
    # ---------- 产品与协作 ----------
    Capability(
        key="docs",
        name="文档撰写与技术整理",
        category="产品与协作",
        keywords=("文档", "手册", "使用手册", "笔记", "总结", "学习总结", "撰写"),
        actions=("把项目沉淀成「README + 设计说明 + 复盘」三件套，形成可展示的文档习惯",),
    ),
    Capability(
        key="tech_watch",
        name="前沿跟进与技术调研",
        category="产品与协作",
        keywords=("论文", "前沿", "技术调研", "开源框架", "调研主流", "新技术", "技术方案"),
        actions=("每周一篇技术调研简报（选型对比 + 落地建议），持续 4 周就有素材可讲",),
    ),
    Capability(
        key="product_research",
        name="行业 / 用户 / 竞品调研",
        category="产品与协作",
        keywords=("行业调研", "竞品", "用户需求", "需求梳理", "市场", "用户研究"),
        actions=("做一份 AI 产品竞品矩阵（功能 / 定价 / 目标用户 + 结论），面试直接可用",),
    ),
    Capability(
        key="prd",
        name="PRD / 需求文档与原型",
        category="产品与协作",
        keywords=("prd", "需求文档", "原型", "交互逻辑", "功能流程", "axure", "墨刀", "功能说明", "功能原型"),
        subs=(
            _sub("需求文档", "prd", "需求文档", "功能流程", "交互逻辑"),
            _sub("原型工具", "axure", "墨刀", "原型"),
        ),
        actions=("用墨刀 / Axure 仿写一份 AI 助手 PRD（含功能流程与交互说明）",),
    ),
    Capability(
        key="data_analysis",
        name="数据分析（Excel / SQL）",
        category="产品与协作",
        keywords=("excel", "sql", "数据分析", "数据复盘", "核心数据", "数据看板"),
        subs=(_sub("Excel", "excel", "表格", "数据透视", "图表"), _sub("SQL", "sql", "数据库查询", "查询语句")),
        actions=("补 SQL 基础（增删改查 + 分组聚合），并用 Excel 做一次真实数据复盘",),
    ),
    Capability(
        key="cross_team",
        name="跨团队沟通与对接",
        category="产品与协作",
        keywords=("沟通", "对接", "协作", "跨团队", "配合", "同步需求", "协调", "跨部门"),
        actions=("准备 1 个跨角色协作的 STAR 案例（背景 → 我做了什么 → 结果）",),
    ),
    Capability(
        key="user_feedback",
        name="用户反馈与迭代",
        category="产品与协作",
        keywords=("用户反馈", "反馈", "迭代", "体验优化", "测试验收", "验收"),
        actions=("参与一次完整反馈闭环（收集 → 归因 → 改动 → 验证），记录前后对比",),
    ),
    # ---------- 通用素质 ----------
    Capability(
        key="learning",
        name="学习能力与主动性",
        category="通用素质",
        keywords=(
            "学习能力", "主动学习", "快速上手", "快速接收", "主动", "踏实", "耐心", "细致",
            "执行力", "责任心", "自驱",
        ),
        actions=("简历里用事实替换形容词：新任务 3 天上手 → 交付了什么结果",),
    ),
    Capability(
        key="ai_interest",
        name="AI 行业兴趣与产品认知",
        category="通用素质",
        keywords=("浓厚兴趣", "兴趣", "行业知识", "ai工具", "深度体验", "ai产品"),
        actions=("写一份「我用 AI 工具做过什么」清单，展示真实使用深度而不是「感兴趣」",),
    ),
    # ---------- 加分经历 ----------
    Capability(
        key="competition",
        name="竞赛 / 科研 / 课程项目经历",
        category="加分经历",
        keywords=("竞赛", "kaggle", "数模", "科研", "课程项目", "课程作业", "比赛", "获奖", "项目经验"),
        actions=("挑最相关的 1-2 个竞赛或科研，按「问题 - 方法 - 结果 - 量化数字」重写",),
    ),
    Capability(
        key="open_source",
        name="开源 / 个人作品",
        category="加分经历",
        keywords=("开源", "个人作品", "个人项目", "小项目", "github", "作品集", "demo", "个人ai小项目"),
        actions=("整理 1-2 个可展示项目（README + 截图 / demo + 一键运行），放在简历最显眼处",),
    ),
)

CAPABILITY_BY_KEY = {cap.key: cap for cap in CAPABILITIES}

# ---- 关键词匹配 -------------------------------------------------------------

_ASCII_CACHE = {}


def _ascii_pattern(keyword: str):
    pattern = _ASCII_CACHE.get(keyword)
    if pattern is None:
        pattern = re.compile(r"(?<![a-z0-9])" + re.escape(keyword) + r"(?![a-z0-9])")
        _ASCII_CACHE[keyword] = pattern
    return pattern


def find_keyword(text_lower: str, keyword: str):
    """返回关键词在文本中的起始位置，找不到返回 -1。

    纯英文关键词按单词边界匹配（避免 rag 命中 storage、git 命中 github）。
    """
    kw = keyword.lower()
    if not kw:
        return -1
    if kw.isascii():
        match = _ascii_pattern(kw).search(text_lower)
        return match.start() if match else -1
    return text_lower.find(kw)


def kw_hit(text_lower: str, keyword: str) -> bool:
    return find_keyword(text_lower, keyword) >= 0


# 否定词：命中这些前缀时，该关键词不计入「已具备」证据
NEGATION_PREFIXES = ("未", "没", "无", "尚未", "不会", "没有", "缺乏", "欠缺", "零基础", "不熟", "待学")
CONTEXT_WINDOW = 24
CLAUSE_BOUNDARIES = "，。；、！？,;.!?（）()【】<>：:\n"


def clause_context(text_lower: str, position: int, window: int = CONTEXT_WINDOW) -> str:
    """取关键词前的一小段上下文，遇到标点就截断（否定/弱化的作用域通常在一个小句内）。"""
    start = max(0, position - window)
    context = text_lower[start:position]
    cut = max(context.rfind(char) for char in CLAUSE_BOUNDARIES) if context else -1
    return context[cut + 1:]


def is_negated(text_lower: str, position: int) -> bool:
    context = clause_context(text_lower, position)
    return any(prefix in context for prefix in NEGATION_PREFIXES)


def _is_negated(text_lower: str, position: int) -> bool:  # 兼容旧调用
    return is_negated(text_lower, position)


def keyword_hits(text_lower: str, keyword: str) -> List[int]:
    """返回关键词所有「未被否定」的出现位置。"""
    kw = keyword.lower()
    positions: List[int] = []
    if not kw:
        return positions
    if kw.isascii():
        positions = [match.start() for match in _ascii_pattern(kw).finditer(text_lower)]
    else:
        start = 0
        while True:
            found = text_lower.find(kw, start)
            if found < 0:
                break
            positions.append(found)
            start = found + len(kw)
    return [pos for pos in positions if not is_negated(text_lower, pos)]


def match_capabilities(text: str, negation: bool = False) -> List[str]:
    """把一段文本映射成能力 key 列表。

    negation=True 时用于项目描述：写了「未接触过向量数据库」，不会算成 RAG 的证据。
    """
    text_lower = text.lower()
    matched: List[str] = []
    for cap in CAPABILITIES:
        hit = False
        for keyword in cap.all_keywords:
            position = find_keyword(text_lower, keyword)
            if position < 0:
                continue
            if negation and _is_negated(text_lower, position):
                continue
            hit = True
            break
        if hit:
            matched.append(cap.key)
    return matched


def match_sub_items(text: str, subs: Iterable[SubItem], negation: bool = False) -> List[str]:
    """返回文本命中的子维度名称。"""
    text_lower = text.lower()
    hits: List[str] = []
    for sub in subs:
        for keyword in sub.keywords:
            position = find_keyword(text_lower, keyword)
            if position < 0:
                continue
            if negation and _is_negated(text_lower, position):
                continue
            hits.append(sub.name)
            break
    return hits
