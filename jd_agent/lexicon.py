"""能力词典：把 JD 与项目描述里的自然语言，映射成可对比的能力项。

这一层是「离线可用、可检查」的关键：不依赖大模型也能稳定抽出「岗位需要什么能力」。
想扩展能力项时，直接往 CAPABILITIES 里加一条 Capability 即可。

v3 扩充：
- 新增 Agent / 多模态生成 / LLM 工程化 / 评测 / 向量库 / 数据工程 / 云原生 /
  CV / NLP / RL / AI 产品 / 数据驱动 共 12 项
- 加固现有能力项的同义词
- 扩充否定词表并按长度倒序匹配
"""
from __future__ import annotations

import re
from typing import Iterable, List, Tuple

from .schema import Capability, SubItem


def _sub(name: str, *keywords: str) -> SubItem:
    return SubItem(name=name, keywords=tuple(keywords))


# ============================================================
# 同义词表：匹配前把「变体」归一化成「标准词」
# 格式：{标准词: (变体1, 变体2, ...)}
# ============================================================
SYNONYMS: dict[str, tuple[str, ...]] = {
    # 程度副词归一化
    "熟练": ("精熟", "熟稔", "精通", "熟练掌握", "娴熟", "很熟"),
    "掌握": ("会", "能", "具备", "拥有", "懂"),
    "了解": ("知晓", "知道", "清楚", "明白", "有所了解"),

    # 动作词归一化
    "搭建": ("构建", "搭建", "搭起", "建成", "组建", "开发"),
    "实现": ("完成", "做出", "落地", "开发", "编写"),
    "调优": ("优化", "调优", "改进", "提升", "打磨"),
    "部署": ("上线", "发布", "部署", "release"),

    # 结果词归一化
    "提升": ("提高", "增长", "上涨", "优化了", "增加到"),
    "降低": ("减少", "下降", "缩短", "压缩到", "降低到"),
    "准确率": ("精度", "正确率", "accuracy"),

    # 能力词归一化
    "大模型": ("LLM", "大语言模型", "foundation model", "基础模型"),
    "检索增强": ("RAG", "检索增强生成", "retrieval augmented"),
    "微调": ("finetune", "fine-tune", "指令微调", "SFT"),
    "提示词": ("prompt", "提示工程", "prompt engineering"),
    "智能体": ("Agent", "agent", "AI Agent"),
    "向量库": ("向量数据库", "vector db", "vector database"),
}

def normalize_text(text: str) -> str:
    """把同义词变体替换成标准词，方便后续匹配。"""
    for standard, variants in SYNONYMS.items():
        for variant in variants:
            if variant != standard:
                text = text.replace(variant, standard)
    return text

CAPABILITIES: Tuple[Capability, ...] = (
    # ============================================================
    # 准入门槛
    # ============================================================
    Capability(
        key="education",
        name="学历与专业要求",
        category="准入门槛",
        keywords=("本科", "硕士", "博士", "研究生", "在读", "应届", "学历", "学位",
                  "计算机专业", "相关专业", "专业要求", "不限专业", "统招"),
        subs=(
            _sub("学历层次", "本科", "硕士", "博士", "研究生", "学历", "学位"),
            _sub("专业匹配", "计算机", "人工智能", "软件工程", "数学", "统计",
                 "电子信息", "自动化", "经管", "相关专业"),
        ),
        actions=("若专业不完全对口，用「相关课程 + 项目 + 竞赛」三件事补齐证明，并在简历首行标注学历与专业",),
    ),
    Capability(
        key="cohort",
        name="届别与投递身份",
        category="准入门槛",
        keywords=("25届", "26届", "27届", "28届", "应届", "校招", "秋招", "春招",
                  "在校生", "届别", "招聘对象", "毕业时间", "日常实习", "暑期实习"),
        actions=("在简历首行写明届别与可投身份（校招 / 日常实习 / 暑期实习）",),
    ),
    Capability(
        key="stability",
        name="实习时长与稳定性",
        category="准入门槛",
        keywords=("稳定实习", "长期稳定", "实习时长", "实习时间", "实习期限",
                  "可实习", "连续实习", "实习2个月", "实习三个月", "实习六个月",
                  "出勤", "实习天数", "每周到岗"),
        actions=("明确写清可实习的起止时间与每周天数（例如「2026.03 起，每周 4 天，可连续 6 个月」）",),
    ),

    # ============================================================
    # 硬技能
    # ============================================================
    Capability(
        key="python",
        name="Python 编程能力",
        category="硬技能",
        keywords=("python", "python3", "py脚本", "python脚本", "自动化脚本"),
        actions=("保持手感：每周 2-3 道题 + 一个小脚本沉淀到 GitHub，简历用「用 Python 做过什么」而不是「会 Python」",),
    ),
    Capability(
        key="dl_framework",
        name="深度学习框架（PyTorch / TensorFlow）",
        category="硬技能",
        keywords=("pytorch", "torch", "tensorflow", "keras", "mindspore", "paddle",
                  "paddlepaddle", "深度学习框架", "框架训练", "pytorch lightning"),
        subs=(
            _sub("PyTorch", "pytorch", "torch", "pytorch lightning"),
            _sub("TensorFlow", "tensorflow", "keras"),
            _sub("国产框架", "mindspore", "paddle", "paddlepaddle"),
        ),
        actions=("用 PyTorch 独立复现一个完整训练流程（数据加载 → 训练 → 评估 → 保存），把代码与实验记录开源",),
    ),
    Capability(
        key="vector_db",
        name="向量数据库与检索",
        category="硬技能",
        keywords=("向量数据库", "vector database", "chroma", "milvus", "faiss",
                  "qdrant", "pinecone", "weaviate", "pgvector", "elasticsearch",
                  "opensearch", "embedding", "重排", "rerank", "bge", "m3e",
                  "混合检索", "hybrid search"),
        subs=(
            _sub("向量库产品", "chroma", "milvus", "faiss", "qdrant", "pinecone",
                 "weaviate", "pgvector"),
            _sub("Embedding 模型", "embedding", "bge", "m3e", "text-embedding"),
            _sub("检索优化", "重排", "rerank", "混合检索", "hybrid search"),
        ),
        actions=("对比 2 个向量库（如 Chroma vs Milvus）的检索效果与部署成本，写一份选型对比",),
    ),
    Capability(
        key="data_engineering",
        name="数据工程（ETL / 爬虫 / 标注）",
        category="硬技能",
        keywords=("etl", "数据管道", "pipeline", "爬虫", "scrapy", "selenium",
                  "playwright", "数据标注", "标注平台", "label studio", "dvc",
                  "airflow", "数据治理", "数据质量", "去重", "清洗", "数据采集"),
        subs=(
            _sub("数据采集", "爬虫", "scrapy", "selenium", "playwright", "数据采集"),
            _sub("数据管道", "etl", "pipeline", "airflow", "dvc"),
            _sub("数据标注", "标注", "label studio", "标注平台"),
        ),
        actions=("写一个可复用的数据管道（采集 → 清洗 → 去重 → 入库），附数据质量报告",),
    ),

    # ============================================================
    # 算法基础
    # ============================================================
    Capability(
        key="ml_basics",
        name="机器学习 / 深度学习基础理论",
        category="算法基础",
        keywords=("机器学习", "深度学习", "神经网络", "分类", "回归", "聚类",
                  "算法基础", "模型原理", "监督学习", "无监督", "特征工程",
                  "模型评估", "交叉验证", "过拟合"),
        subs=(
            _sub("机器学习基础", "机器学习", "分类", "回归", "聚类", "监督学习",
                 "无监督", "特征工程", "交叉验证"),
            _sub("深度学习", "深度学习", "神经网络", "cnn", "rnn", "transformer"),
        ),
        actions=("系统复习 ML 基础，产出一份可自测的知识卡片（面试高频：过拟合、评估指标、损失函数）",),
    ),
    Capability(
        key="nlp_basics",
        name="自然语言处理基础",
        category="算法基础",
        keywords=("自然语言处理", "nlp", "文本分类", "命名实体", "ner", "分词",
                  "情感分析", "文本匹配", "句向量", "bert", "roberta", "tokenizer"),
        subs=(
            _sub("文本分类", "文本分类", "情感分析"),
            _sub("序列标注", "命名实体", "ner", "分词"),
            _sub("预训练模型", "bert", "roberta", "tokenizer"),
        ),
        actions=("用 BERT 做一次文本分类微调，记录数据构造与指标变化",),
    ),
    Capability(
        key="cv_basics",
        name="计算机视觉基础",
        category="算法基础",
        keywords=("计算机视觉", "cv", "目标检测", "yolo", "分割", "segmentation",
                  "图像分类", "resnet", "vit", "detr", "ocr", "图像处理"),
        subs=(
            _sub("检测分割", "目标检测", "yolo", "分割", "segmentation", "detr"),
            _sub("图像分类", "图像分类", "resnet", "vit"),
            _sub("OCR", "ocr", "文字识别"),
        ),
        actions=("跑通一个检测或分割的 baseline（如 YOLO），记录指标与调参过程",),
    ),
    Capability(
        key="rl_basics",
        name="强化学习 / RLHF",
        category="算法基础",
        keywords=("强化学习", "rl", "rlhf", "ppo", "dpo", "grpo", "reward model",
                  "奖励模型", "对齐", "alignment"),
        subs=(
            _sub("RL 基础", "强化学习", "ppo", "grpo"),
            _sub("RLHF", "rlhf", "dpo", "reward model", "奖励模型", "对齐"),
        ),
        actions=("用 TRL 跑一次 DPO 或 PPO 小实验，记录对齐前后效果对比",),
    ),

    # ============================================================
    # 后端开发
    # ============================================================
    Capability(
        key="backend_dev",
        name="后端开发（服务端 / 数据库 / 中间件）",
        category="工程实践",
        keywords=(
            # 岗位 / 角色
            "后端开发", "服务端开发", "后端工程师", "backend",
            # 语言 / 框架
            "spring", "spring boot", "springboot", "java", "golang", "go 语言",
            "node.js", "nodejs", "express", "nestjs", "koa",
            "django", "flask", "fastapi", "tornado", "gin", "echo", "fiber",
            # 数据库
            "mysql", "postgresql", "postgres", "oracle", "sql server", "mongodb",
            "redis", "数据库设计", "数据库优化", "索引优化", "慢查询",
            "orm", "sqlalchemy", "mybatis", "gorm", "jpa",
            # 中间件
            "消息队列", "mq", "kafka", "rabbitmq", "rocketmq", "nats",
            "缓存", "cache", "redis", "memcached",
            # 架构与工程
            "微服务", "microservice", "分布式", "distributed", "高并发",
            "高可用", "负载均衡", "restful", "rest api", "grpc",
            "接口设计", "api 设计", "系统设计", "架构设计",
            "鉴权", "认证", "jwt", "oauth", "权限管理",
        ),
        subs=(
            _sub("后端语言", "java", "golang", "go 语言", "node.js", "nodejs", "python"),
            _sub("后端框架", "spring", "spring boot", "springboot", "django",
                 "flask", "fastapi", "express", "nestjs", "gin", "gorm"),
            _sub("关系型数据库", "mysql", "postgresql", "postgres", "oracle",
                 "sql server", "数据库设计", "索引优化"),
            _sub("缓存与队列", "redis", "memcached", "kafka", "rabbitmq",
                 "rocketmq", "消息队列", "缓存"),
            _sub("架构能力", "微服务", "分布式", "高并发", "高可用",
                 "负载均衡", "系统设计", "架构设计"),
        ),
        actions=(
            "用一门后端语言（Java/Go/Python）搭一个完整 CRUD 服务："
            "数据库建模 → 接口设计 → 鉴权 → 缓存 → 部署，"
            "README 写清架构图与接口文档（Swagger），"
            "面试重点讲「为什么这样设计」而不是「用了什么框架」",
        ),
    ),

    # ============================================================
    # 大模型应用
    # ============================================================
    Capability(
        key="llm_basics",
        name="大模型基础原理与认知",
        category="大模型应用",
        keywords=("大模型", "llm", "大语言模型", "生成式", "aigc", "transformer",
                  "预训练", "基础模型", "foundation model", "token",
                  "上下文窗口", "智能问答", "智能产品"),
        actions=("整理一份大模型基础笔记（Transformer / 预训练 / 推理 / 幻觉），能在面试里讲清楚链路",),
    ),
    Capability(
        key="rag",
        name="RAG 与知识库应用",
        category="大模型应用",
        keywords=("rag", "检索增强", "检索增强生成", "知识库", "知识问答",
                  "文档问答", "企业知识库", "客服机器人", "问答系统"),
        subs=(
            _sub("知识库搭建", "知识库", "rag", "检索增强", "问答"),
            _sub("向量检索", "向量数据库", "向量库", "embedding", "faiss",
                 "milvus", "chroma", "检索"),
        ),
        actions=("用 LangChain / LlamaIndex 搭一个最小可用 RAG（切分 → 向量化 → 召回 → 引用）并写 README",),
    ),
    Capability(
        key="finetune",
        name="模型微调",
        category="大模型应用",
        keywords=("微调", "finetune", "fine-tune", "lora", "qlora", "sft",
                  "指令微调", "增量训练", "领域适配", "peft", "adapter"),
        actions=("用 LoRA 微调一个开源小模型，记录数据构造方式与前后效果对比",),
    ),
    Capability(
        key="prompt",
        name="Prompt 工程",
        category="大模型应用",
        keywords=("prompt", "提示词", "提示工程", "prompt engineering",
                  "few-shot", "cot", "思维链", "chain of thought"),
        actions=("沉淀一份 Prompt 对照实验记录（bad case → 改法 → 指标变化）",),
    ),
    Capability(
        key="agent",
        name="Agent 与工具调用",
        category="大模型应用",
        keywords=("agent", "智能体", "function calling", "工具调用", "tool use",
                  "mcp", "model context protocol", "react", "planning",
                  "多智能体", "multi-agent", "autogen", "langgraph", "crewai"),
        subs=(
            _sub("Agent 框架", "langgraph", "autogen", "crewai", "agent", "智能体"),
            _sub("工具调用", "function calling", "工具调用", "tool use", "mcp"),
            _sub("规划与记忆", "planning", "记忆", "memory", "react"),
        ),
        actions=("用 LangGraph 或自研循环搭一个能用 2-3 个工具的 Agent，记录工具调用日志与失败案例",),
    ),
    Capability(
        key="multimodal_gen",
        name="多模态生成（文生图 / 视频 / 语音）",
        category="大模型应用",
        keywords=("文生图", "文生视频", "text-to-image", "text-to-video",
                  "扩散模型", "diffusion", "stable diffusion", "sdxl", "flux",
                  "controlnet", "语音合成", "tts", "语音识别", "asr",
                  "whisper", "声音克隆", "多模态"),
        subs=(
            _sub("图像生成", "文生图", "diffusion", "stable diffusion", "sdxl",
                 "flux", "controlnet"),
            _sub("视频生成", "文生视频", "text-to-video", "sora", "可灵", "runway"),
            _sub("语音", "tts", "asr", "whisper", "语音合成", "语音识别", "声音克隆"),
        ),
        actions=("选一个方向（图/视频/语音）跑通开源模型 + 一次 LoRA 微调，产出对比样例",),
    ),
    Capability(
        key="llm_engineering",
        name="大模型工程化与推理优化",
        category="大模型应用",
        keywords=("vllm", "sglang", "tgi", "推理加速", "量化", "quantization",
                  "gptq", "awq", "gguf", "蒸馏", "distillation", "剪枝",
                  "kv cache", "批处理", "batching", "显存优化", "模型压缩",
                  "onnx", "tensorrt", "模型转换"),
        subs=(
            _sub("推理框架", "vllm", "sglang", "tgi", "tensorrt", "onnx"),
            _sub("量化压缩", "量化", "gptq", "awq", "gguf", "剪枝", "蒸馏"),
            _sub("性能优化", "kv cache", "批处理", "显存优化", "推理加速"),
        ),
        actions=("用 vLLM 部署一个 7B 模型，记录吞吐/延迟对比；再用 GPTQ 量化一次，对比效果与显存",),
    ),
    Capability(
        key="llm_eval",
        name="模型评测与基准测试",
        category="大模型应用",
        keywords=("评测", "benchmark", "评估集", "eval", "mmlu", "ceval",
                  "gsm8k", "人工评估", "llm-as-judge", "ragas", "bleu",
                  "rouge", "perplexity"),
        subs=(
            _sub("自动评测", "benchmark", "mmlu", "ceval", "gsm8k", "ragas"),
            _sub("人工评估", "人工评估", "llm-as-judge", "标注评分"),
        ),
        actions=("为一个 RAG 或微调项目建一套评测集（20-50 条），记录改动前后指标变化",),
    ),

    # ============================================================
    # 工程实践
    # ============================================================
    Capability(
        key="web_api",
        name="接口 / Web 后端开发",
        category="工程实践",
        keywords=("fastapi", "flask", "django", "tornado", "web开发", "接口开发",
                  "接口联调", "封装基础模型接口", "后端", "服务端", "api",
                  "restful", "rest api", "grpc", "swagger", "接口文档"),
        actions=("用 FastAPI 把模型或 RAG 流程封装成 HTTP 接口（带流式输出更好）",),
    ),
    Capability(
        key="debug_eng",
        name="调试与问题排查",
        category="工程实践",
        keywords=("调试", "bug", "排查", "联调", "测试验证", "问题定位",
                  "定位问题", "问题清单", "排错", "异常处理", "日志分析",
                  "性能分析", "profiling"),
        actions=("留一份排障记录：现象 → 日志 → 定位 → 修复 → 回归，面试可直接讲",),
    ),
    Capability(
        key="data_process",
        name="数据处理与清洗",
        category="工程实践",
        keywords=("数据处理", "数据清洗", "样本筛选", "数据集", "数据整理",
                  "数据质量", "标注", "去重", "格式统一"),
        actions=("写一个可复用清洗脚本（去重 / 过滤 / 格式统一 / 输出统计报告）",),
    ),
    Capability(
        key="experiment",
        name="实验与效果评估复盘",
        category="工程实践",
        keywords=("效果评估", "实验", "复盘", "准确率", "调优", "参数调优",
                  "实验日志", "结果对比", "指标", "消融", "ablation",
                  "对比实验", "超参搜索"),
        actions=("建立实验记录模板（配置 / 指标 / 结论），把一次真实调优过程写成可展示的材料",),
    ),
    Capability(
        key="tools_git_linux",
        name="Linux / Git 工程工具",
        category="工程实践",
        keywords=("linux", "git", "版本管理", "命令行", "bash", "shell",
                  "vim", "ssh", "github", "gitlab", "gitee"),
        subs=(
            _sub("Linux", "linux", "命令行", "bash", "shell", "vim", "ssh"),
            _sub("Git", "git", "版本管理", "github", "gitlab", "gitee"),
        ),
        actions=("完整走一遍 Git 协作流程（分支 / PR / 冲突解决），并熟悉 Linux 常用命令",),
    ),
    Capability(
        key="docker",
        name="Docker 与部署",
        category="工程实践",
        keywords=("docker", "dockerfile", "docker-compose", "容器", "镜像",
                  "容器化", "部署", "上线"),
        subs=(
            _sub("Docker", "docker", "dockerfile", "docker-compose", "容器", "镜像"),
            _sub("部署上线", "部署", "上线"),
        ),
        actions=("把个人项目打成镜像并在云服务器上跑通，README 附部署步骤",),
    ),
    Capability(
        key="cloud_native",
        name="云原生与 CI/CD",
        category="工程实践",
        keywords=("kubernetes", "k8s", "ci/cd", "github actions", "gitlab ci",
                  "jenkins", "阿里云", "腾讯云", "aws", "gcp", "azure",
                  "serverless", "负载均衡", "nginx", "监控", "prometheus"),
        subs=(
            _sub("容器编排", "kubernetes", "k8s"),
            _sub("CI/CD", "ci/cd", "github actions", "gitlab ci", "jenkins"),
            _sub("云服务", "阿里云", "腾讯云", "aws", "gcp", "azure", "serverless"),
        ),
        actions=("给个人项目加一个 GitHub Actions 自动测试 + 部署流程，README 附上流程说明",),
    ),
    Capability(
        key="frontend",
        name="前端 / 全栈能力",
        category="工程实践",
        keywords=("前端", "全栈", "页面", "小程序", "web页面", "可视化页面",
                  "react", "vue", "html", "css", "javascript", "typescript"),
        actions=("补基础前端：能改页面、能接接口、能部署静态站点",),
    ),

    # ============================================================
    # 产品与协作
    # ============================================================
    Capability(
        key="ai_product",
        name="AI 产品设计与落地",
        category="产品与协作",
        keywords=("ai产品", "大模型产品", "ai应用", "产品落地", "场景落地",
                  "用户旅程", "ai交互", "对话设计", "prompt 设计", "ai 工作流"),
        subs=(
            _sub("AI 产品设计", "ai产品", "ai应用", "对话设计", "ai交互"),
            _sub("落地能力", "场景落地", "产品落地", "ai 工作流"),
        ),
        actions=("拆解 3 个 AI 产品的核心链路（输入 → 模型 → 输出 → 反馈），写一份对比笔记",),
    ),
    Capability(
        key="data_driven",
        name="数据驱动决策",
        category="产品与协作",
        keywords=("数据驱动", "ab test", "abtest", "埋点", "漏斗", "留存",
                  "转化率", "北极星指标", "数据看板", "增长"),
        subs=(
            _sub("实验方法", "ab test", "abtest", "漏斗", "留存"),
            _sub("指标体系", "北极星指标", "数据看板", "转化率"),
        ),
        actions=("给一个真实项目定义 3 个核心指标 + 1 个北极星指标，并设计一次 AB 实验",),
    ),
    Capability(
        key="docs",
        name="文档撰写与技术整理",
        category="产品与协作",
        keywords=("文档", "readme", "技术文档", "设计文档", "手册", "使用手册",
                  "笔记", "总结", "学习总结", "撰写", "wiki", "notion"),
        actions=("把项目沉淀成「README + 设计说明 + 复盘」三件套，形成可展示的文档习惯",),
    ),
    Capability(
        key="tech_watch",
        name="前沿跟进与技术调研",
        category="产品与协作",
        keywords=("论文", "论文复现", "前沿", "技术调研", "开源框架",
                  "调研主流", "新技术", "技术方案", "arxiv", "顶会",
                  "技术博客", "开源社区"),
        actions=("每周一篇技术调研简报（选型对比 + 落地建议），持续 4 周就有素材可讲",),
    ),
    Capability(
        key="product_research",
        name="行业 / 用户 / 竞品调研",
        category="产品与协作",
        keywords=("行业调研", "竞品", "竞品分析", "用户需求", "需求梳理",
                  "市场", "用户研究", "用户调研"),
        actions=("做一份 AI 产品竞品矩阵（功能 / 定价 / 目标用户 + 结论），面试直接可用",),
    ),
    Capability(
        key="prd",
        name="PRD / 需求文档与原型",
        category="产品与协作",
        keywords=("prd", "需求文档", "原型", "交互逻辑", "功能流程",
                  "axure", "墨刀", "功能说明", "功能原型"),
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
        keywords=("excel", "sql", "数据分析", "数据复盘", "核心数据",
                  "数据看板", "数据透视", "查询语句"),
        subs=(
            _sub("Excel", "excel", "表格", "数据透视", "图表"),
            _sub("SQL", "sql", "数据库查询", "查询语句"),
        ),
        actions=("补 SQL 基础（增删改查 + 分组聚合），并用 Excel 做一次真实数据复盘",),
    ),
    Capability(
        key="cross_team",
        name="跨团队沟通与对接",
        category="产品与协作",
        keywords=("跨团队", "跨部门", "团队协作", "沟通协调", "对接协作",
                  "同步需求", "协调资源", "配合团队", "协作能力"),
        actions=("准备 1 个跨角色协作的 STAR 案例（背景 → 我做了什么 → 结果）",),
    ),
    Capability(
        key="user_feedback",
        name="用户反馈与迭代",
        category="产品与协作",
        keywords=("用户反馈", "反馈迭代", "迭代优化", "体验优化",
                  "测试验收", "验收", "用户回访"),
        actions=("参与一次完整反馈闭环（收集 → 归因 → 改动 → 验证），记录前后对比",),
    ),

    # ============================================================
    # 通用素质
    # ============================================================
    Capability(
        key="learning",
        name="学习能力与主动性",
        category="通用素质",
        keywords=("学习能力", "主动学习", "快速上手", "快速接收", "主动",
                  "踏实", "耐心", "细致", "执行力", "责任心", "自驱",
                  "自学", "抗压", "适应能力"),
        actions=("简历里用事实替换形容词：新任务 3 天上手 → 交付了什么结果",),
    ),
    Capability(
        key="ai_interest",
        name="AI 行业兴趣与产品认知",
        category="通用素质",
        keywords=("浓厚兴趣", "兴趣", "行业知识", "ai工具", "深度体验",
                  "ai产品", "chatgpt", "claude", "cursor", "copilot"),
        actions=("写一份「我用 AI 工具做过什么」清单，展示真实使用深度而不是「感兴趣」",),
    ),

    # ============================================================
    # 加分经历
    # ============================================================
    Capability(
        key="competition",
        name="竞赛 / 科研 / 课程项目经历",
        category="加分经历",
        keywords=("竞赛", "kaggle", "天池", "数模", "数学建模", "acm",
                  "蓝桥杯", "挑战杯", "互联网+", "科研", "课程项目",
                  "课程作业", "比赛", "获奖", "项目经验"),
        actions=("挑最相关的 1-2 个竞赛或科研，按「问题 - 方法 - 结果 - 量化数字」重写",),
    ),
    Capability(
        key="open_source",
        name="开源 / 个人作品",
        category="加分经历",
        keywords=("开源", "开源贡献", "个人作品", "个人项目", "小项目",
                  "github", "github 项目", "star", "pr", "作品集",
                  "demo", "个人ai小项目"),
        actions=("整理 1-2 个可展示项目（README + 截图 / demo + 一键运行），放在简历最显眼处",),
    ),
)

CAPABILITY_BY_KEY = {cap.key: cap for cap in CAPABILITIES}

# ============================================================
# 关键词匹配
# ============================================================

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


# ------------------------------------------------------------
# 否定词表（v3 扩充）
# 注意：按长度倒序匹配，避免「未」先命中「未接触」
# ------------------------------------------------------------
NEGATION_PREFIXES = (
    # 中文否定
    "未", "没", "无", "尚未", "不会", "没有", "缺乏", "欠缺", "零基础", "待学",
    "不熟", "不熟悉", "不了解", "没接触", "未接触", "没做过", "未做过",
    "没经验", "无经验", "零经验", "没实践", "未实践", "没落地", "未落地",
    "非", "不涉及", "不参与", "没参与", "未参与", "不是", "不负责", "未负责",
    "待补", "待加强", "待提升", "还不会", "尚未掌握", "未掌握", "没掌握",
    "不精", "不擅长", "不熟练", "待熟悉",
    # 英文否定
    "no ", "not ", "never ", "without ", "lack of ", "not familiar",
)

# 按长度倒序，确保长词优先匹配
NEGATION_PREFIXES_SORTED = tuple(sorted(NEGATION_PREFIXES, key=len, reverse=True))

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
    return any(prefix in context for prefix in NEGATION_PREFIXES_SORTED)


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
    text = normalize_text(text)
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
    text = normalize_text(text)
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