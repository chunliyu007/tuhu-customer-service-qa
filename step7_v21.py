#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step7_v21.py

里程碑 Milestone 7：全局反思与 HTML 报告全面升级（V2.1）。

功能：
1) 先对全量 Session 执行 Map 阶段（V2.1 严格：common_sense_reason + prompt_defect_suspected/defect_reason）。
2) Reduce 阶段：收集 prompt_defect_suspected=true 的坏案子，拼接成全局反思输入文本，
   再进行最后一次大模型调用，输出《全局系统 Prompt 迭代建议》Markdown。
3) 报告生成：使用 pyecharts 生成柱状图/饼图等，并在 HTML 中插入「一级类目占比 + 合并二级类目（含说明列）」表格，
   把 Reduce Markdown 转 HTML 并插入底部，输出 tu_hu_analysis_report_v21.html。

注意：
- 为避免一次性上下文爆炸，坏案文本会做长度截断（可配置）。
- 默认会在本地落盘 Map JSON 与 Reduce Markdown，便于你反复打开/二次生成报告。
"""

import argparse
from collections import Counter
import json
import os
import re
import threading
import html as html_lib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

import openai
import pandas as pd
from openai import OpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

import step1
import step2
import step5_v21
import step6_v21

from pyecharts import options as opts
from pyecharts.charts import Bar, Page, Pie
from pyecharts.globals import ThemeType

import markdown as md_lib


def ensure_nonempty_str(x: Any) -> str:
    s = "" if x is None else str(x)
    return s


def sanitize_llm_output_as_json(text: str) -> str:
    """
    将 LLM 输出尽量清洗成“单个 JSON 对象字符串”。
    """
    text = (text or "").strip()
    # 去掉可能的 ```json ... ``` 外壳（如果模型输出了）
    code_block_pattern = re.compile(
        r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE
    )
    m = code_block_pattern.match(text)
    if m:
        text = m.group(1).strip()

    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()
    return text


def count_chinese_chars(s: str) -> int:
    return len(re.findall(r"[\u4e00-\u9fff]", s or ""))


def validate_secondary_category(sec: Any) -> None:
    # 只做提示，不强行改写
    sec_str = ensure_nonempty_str(sec)
    if count_chinese_chars(sec_str) > 4:
        print(f"警告：secondary_category 中文字符数 > 4：{sec_str} count={count_chinese_chars(sec_str)}")


def make_retry_decorator(max_attempts: int = 3):
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        retry=retry_if_exception_type(
            (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.APIError,
            )
        ),
    )


def call_llm_for_map(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    user_content: str,
    max_retries: int = 3,
) -> str:
    @make_retry_decorator(max_attempts=max_retries)
    def _call() -> str:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        return resp.choices[0].message.content or ""

    return _call()


def load_input_and_group_sessions(input_path: str) -> Tuple[Any, Any]:
    df_raw = step1.read_input_file(input_path)
    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    return df, session_df


def detect_columns(df_raw) -> Tuple[str, Optional[str], str, Optional[str], Optional[str], Optional[str]]:
    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    tools_col = step1.detect_column(columns, step1.POSSIBLE_TOOLS_ANSWER_COLUMNS)
    if not tools_col:
        raise ValueError("无法识别 toolsmodelanswer 字段，请检查表结构。")
    dialogue_col = step1.detect_column(columns, step1.POSSIBLE_DIALOGUE_SENTENCE_COLUMNS)
    rewrite_col = step1.detect_column(columns, step1.POSSIBLE_REWRITE_QUERY_COLUMNS)
    answer_col = step1.detect_column(columns, step1.POSSIBLE_ANSWER_COLUMNS)
    return session_col, user_col, tools_col, dialogue_col, rewrite_col, answer_col


def build_map_user_content(context_prompt: str) -> str:
    # 严格要求模型输出 V2.1 规范 JSON
    return "请基于下面内容完成微观审查，并输出 V2.1 规范 3 的 JSON：\n\n" + context_prompt + "\n"


def run_map_stage(
    input_path: str,
    mapping_xlsx_path: str,
    strategies_dir: str,
    persona1_docx: str,
    persona2_docx: str,
    map_output_json: str,
    max_sessions: Optional[int],
    max_workers: int,
    max_retries: int,
    force_map: bool,
) -> List[Dict[str, Any]]:
    if os.path.exists(map_output_json) and not force_map:
        print(f"Map 结果已存在，跳过 Map 阶段：{map_output_json}")
        with open(map_output_json, "r", encoding="utf-8") as f:
            return json.load(f)

    print("开始执行 Map 阶段（V2.1 全量/前 N）……")

    df_raw = step1.read_input_file(input_path)
    session_col, user_col, tools_col, dialogue_col, rewrite_col, answer_col = detect_columns(df_raw)

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)

    total_sessions = len(session_df)
    if total_sessions == 0:
        raise ValueError("没有可用 session 数据。")

    if max_sessions is None:
        run_n = total_sessions
    else:
        run_n = max(1, min(max_sessions, total_sessions))

    print(f"Map 阶段将处理 Session 数量：{run_n}（总计 {total_sessions}）")

    mapping_rows = step5_v21.load_intent_strategy_mapping(mapping_xlsx_path)
    persona1_text = step5_v21.read_doc_or_text_plain_text(persona1_docx)
    persona2_text = step5_v21.read_doc_or_text_plain_text(persona2_docx)

    # LLM
    client = OpenAI(api_key=step2.API_KEY, base_url=step2.API_BASE_URL)
    model_name = step2.MODEL_NAME
    system_prompt = step6_v21.build_system_prompt_v21()

    # doc 读取缓存
    doc_cache: Dict[str, str] = {}
    doc_lock = threading.Lock()

    def get_doc_text(path: str) -> str:
        with doc_lock:
            if path in doc_cache:
                return doc_cache[path]
        t = step5_v21.read_doc_or_text_plain_text(path)
        with doc_lock:
            doc_cache[path] = t
        return t

    # 对每个 Session 的分析函数
    semaphore = threading.Semaphore(max_workers)

    def analyze_one(idx: int) -> Dict[str, Any]:
        session_record = session_df.iloc[idx].to_dict()
        row_indices = session_record.get("row_indices") or []
        if not row_indices:
            return {
                "user_id": session_record.get("user_id"),
                "session_id": session_record.get("session_id"),
                "llm_result": None,
                "error": "empty_row_indices",
            }

        # 动态选择专家策略
        (
            best_mapping,
            best_score,
            tool_names,
            _current_tools_text,
            _current_dialogue,
            _current_rewrite,
            _current_answer,
            expert_docx_path,
        ) = step6_v21.choose_expert_strategy_for_session(
            df=df,
            row_indices=row_indices,
            tools_col=tools_col,
            dialogue_col=dialogue_col,
            rewrite_col=rewrite_col,
            answer_col=answer_col,
            mapping_rows=mapping_rows,
            strategies_dir=strategies_dir,
        )

        if best_mapping is None or expert_docx_path is None:
            return {
                "user_id": session_record.get("user_id"),
                "session_id": session_record.get("session_id"),
                "tool_names": sorted(list(tool_names)) if tool_names else [],
                "mounted_expert_docx": None,
                "mapping_best_score": best_score,
                "llm_result": None,
                "error": "mapping_failed",
                "session_history_text": "",
            }

        conversation_rows = step5_v21.extract_conversation_rows_for_session(
            df=df,
            row_indices=row_indices,
            dialogue_col=dialogue_col,
            rewrite_col=rewrite_col,
            answer_col=answer_col,
        )
        session_history_text = step5_v21.build_session_history_text(conversation_rows)
        expert_strategy_text = get_doc_text(expert_docx_path)
        context_prompt = step5_v21.build_context_prompt(
            session_history_text=session_history_text,
            persona1_text=persona1_text,
            persona2_text=persona2_text,
            expert_strategy_text=expert_strategy_text,
        )
        user_content = build_map_user_content(context_prompt)

        try:
            with semaphore:
                llm_raw = call_llm_for_map(
                    client=client,
                    model_name=model_name,
                    system_prompt=system_prompt,
                    user_content=user_content,
                    max_retries=max_retries,
                )
            llm_sanitized = sanitize_llm_output_as_json(llm_raw)  # type: ignore
            parsed = json.loads(llm_sanitized)
        except Exception as exc:
            parsed = {
                "rewrite_issue": "不合理",
                "answer_issue": "不合理",
                "primary_category": "其他",
                "secondary_category": "错误结果",
                "common_sense_reason": "JSON 解析失败/调用失败：请人工检查。",
                "prompt_defect_suspected": False,
                "defect_reason": "",
                "is_typical": False,
            }
            llm_raw = ""

        validate_secondary_category(parsed.get("secondary_category"))

        return {
            "user_id": session_record.get("user_id"),
            "session_id": session_record.get("session_id"),
            "tool_names": sorted(list(tool_names)) if tool_names else [],
            "mounted_expert_docx": expert_docx_path,
            "mapping_best_score": best_score,
            "mapping_intent_desc": best_mapping.intent_desc,
            "session_history_text": session_history_text,
            "llm_result": parsed,
            "error": None,
        }

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_one, i) for i in range(run_n)]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Map进度"):
            results.append(future.result())

    try:
        with open(map_output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Map 结果已保存到：{map_output_json}")
    except Exception as exc:
        print(f"警告：保存 Map 结果失败：{exc}")

    return results


def build_reduce_prompt(
    persona1_text: str,
    persona2_text: str,
    expert_docs: Dict[str, str],
    bad_cases: List[Dict[str, Any]],
    max_bad_cases_chars: int,
) -> str:
    # 坏案拼接（带截断）
    parts: List[str] = []
    total = 0
    for item in bad_cases:
        session_id = item.get("session_id")
        user_id = item.get("user_id")
        defect_reason = ensure_nonempty_str((item.get("llm_result") or {}).get("defect_reason"))
        cs_reason = ensure_nonempty_str((item.get("llm_result") or {}).get("common_sense_reason"))
        history = ensure_nonempty_str(item.get("session_history_text"))

        snippet = (
            f"---\n用户ID: {user_id}\nSessionID: {session_id}\n"
            f"common_sense_reason: {cs_reason}\n"
            f"defect_reason: {defect_reason}\n"
            f"原始对话历史（节选）：\n{history[:1500]}\n"
        )
        if total + len(snippet) > max_bad_cases_chars:
            break
        parts.append(snippet)
        total += len(snippet)

    bad_cases_text = "\n".join(parts)

    expert_docs_text = []
    for fname, content in expert_docs.items():
        expert_docs_text.append(f"【专家策略文档：{fname}】\n{content[:6000]}\n")

    return (
        "你是途虎养车系统级 Prompt 迭代顾问。"
        "请基于以下坏案（客服回答体验失败 + 规则疑点），结合【人设1】、【人设2】和涉及的【专家策略文档】，"
        "输出《全局系统 Prompt 迭代建议》。\n\n"
        "要求（必须遵守）：\n"
        "1) 只输出 Markdown，不要输出任何代码块包裹内容，不要输出 HTML。\n"
        "2) 必须按以下结构输出：\n"
        "   - ## 人设1（Router）修改建议\n"
        "   - ## 人设2（Answerer）修改建议\n"
        "   - ## 各专家策略（按文档分别）修改建议\n"
        "3) 每一条建议都要：指出“建议新增/修改的规则点”+“建议触发的输入特征/场景”+“对本数据集中坏案的预期修复效果”。\n\n"
        f"【人设1】\n{persona1_text}\n\n"
        f"【人设2】\n{persona2_text}\n\n"
        + "\n".join(expert_docs_text)
        + "\n【坏案集合】\n"
        + bad_cases_text
    )


def reduce_global_reflection(
    map_results: List[Dict[str, Any]],
    persona1_docx: str,
    persona2_docx: str,
    strategies_dir: str,
    reduce_output_md: str,
    model_name: str,
    force_reduce: bool,
    max_bad_cases_chars: int,
) -> str:
    if os.path.exists(reduce_output_md) and not force_reduce:
        print(f"Reduce Markdown 已存在，跳过：{reduce_output_md}")
        with open(reduce_output_md, "r", encoding="utf-8") as f:
            return f.read()

    # 收集坏案
    bad_cases: List[Dict[str, Any]] = []
    involved_doc_paths: Set[str] = set()
    for item in map_results:
        llm_result = item.get("llm_result") or {}
        if llm_result.get("prompt_defect_suspected") is True:
            bad_cases.append(item)
            doc = item.get("mounted_expert_docx")
            if doc:
                involved_doc_paths.add(doc)

    print(f"Reduce 阶段坏案数量：{len(bad_cases)}，涉及专家策略 docx 数量：{len(involved_doc_paths)}")
    if not bad_cases:
        raise ValueError("没有坏案（prompt_defect_suspected=true），无法进行 Reduce。")

    persona1_text = step5_v21.read_doc_or_text_plain_text(persona1_docx)
    persona2_text = step5_v21.read_doc_or_text_plain_text(persona2_docx)

    expert_docs: Dict[str, str] = {}
    for doc_path in involved_doc_paths:
        try:
            fname = os.path.basename(doc_path)
            expert_docs[fname] = step5_v21.read_doc_or_text_plain_text(doc_path)
        except Exception as exc:
            print(f"警告：读取专家策略文档失败 {doc_path}：{exc}")

    # 大模型调用
    client = OpenAI(api_key=step2.API_KEY, base_url=step2.API_BASE_URL)
    system_prompt = "你是一个只会输出 Markdown 的写作助手。"
    user_prompt = build_reduce_prompt(
        persona1_text=persona1_text,
        persona2_text=persona2_text,
        expert_docs=expert_docs,
        bad_cases=bad_cases,
        max_bad_cases_chars=max_bad_cases_chars,
    )

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    md = resp.choices[0].message.content or ""

    try:
        with open(reduce_output_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Reduce Markdown 已保存到：{reduce_output_md}")
    except Exception as exc:
        print(f"警告：保存 Reduce Markdown 失败：{exc}")

    return md


def build_sunburst_data(map_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    counts: Dict[str, Dict[str, int]] = {}
    for item in map_results:
        llm = item.get("llm_result") or {}
        primary = ensure_nonempty_str(llm.get("primary_category") or "其他")
        secondary = ensure_nonempty_str(llm.get("secondary_category") or "其他")
        counts.setdefault(primary, {})
        counts[primary][secondary] = counts[primary].get(secondary, 0) + 1

    data: List[Dict[str, Any]] = []
    for primary, sec_dict in counts.items():
        children = [{"name": sec, "value": v} for sec, v in sec_dict.items()]
        data.append({"name": primary, "children": children})
    return data


def truncate_text(s: str, max_len: int = 200) -> str:
    s = ensure_nonempty_str(s)
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


# 二级类目语义合并：按「子串命中」顺序匹配，先匹配的优先（避免「价格对比」被误判为纯对比类）
# 元组含义：(关键词元组, 合并后名称, 简短说明)
SECONDARY_MERGE_GROUPS: List[Tuple[Tuple[str, ...], str, str]] = [
    (
        (
            "异响",
            "故障",
            "诊断",
            "功能异常",
            "全面检测",
            "免费检测",
            "全车检测",
            "专项检查",
            "项目检测",
            "基础检查",
            "车辆检查",
            "检测服务",
        ),
        "故障与异常",
        "车况异响、故障判断、检测类问题。",
    ),
    (
        ("混加", "适配", "车型适配", "接口适配", "兼容", "粘度", "报警线"),
        "适配与兼容",
        "是否适配本车、粘度/规格、混加或配件匹配核对。",
    ),
    (
        (
            "优惠券",
            "赠品",
            "权益",
            "活动",
            "秒杀",
            "优惠查询",
            "优惠",
            "券效期",
            "卡券",
        ),
        "活动与权益",
        "活动价、优惠券、券效期、卡券、赠品或会员权益等。",
    ),
    (
        ("工时", "费用", "价格", "核算"),
        "价格与费用",
        "报价、工时费、费用构成或价格相关解释。",
    ),
    (
        ("缺货", "现货", "库存", "补发", "数量确认"),
        "库存与货源",
        "有无货、现货、库存、补发或数量确认。",
    ),
    (
        ("联系门店", "门店查询", "门店查找", "地址查询", "预约", "门店", "到店"),
        "门店与到店",
        "找店、门店信息、地址、预约到店或到店相关。",
    ),
    (
        ("清洗服务", "安装服务", "清洗方式", "清洁消毒", "施工方式"),
        "到店服务",
        "到店安装、清洗、清洁或施工方式类服务。",
    ),
    (
        (
            "更换周期",
            "更换建议",
            "更换指导",
            "换油方式",
            "周期提醒",
            "紧急补油",
            "补油",
            "滤芯",
            "用量",
            "周期",
            "更换",
        ),
        "用量与周期",
        "用量、保养/更换周期、换油或更换时机。",
    ),
    (
        (
            "自行安装",
            "拆卸",
            "操作指引",
            "安装指导",
            "安装时效",
            "安装咨询",
            "安装确认",
            "抽真空",
        ),
        "安装与操作",
        "自行安装/拆卸步骤、操作指引、安装确认、抽真空或安装时效。",
    ),
    (
        ("订单合并", "项目下单", "售后进度", "售后赔偿", "订单", "下单"),
        "订单与进度",
        "下单、订单状态、合并、售后进度/赔偿或时效。",
    ),
    (
        (
            "套餐推荐",
            "套餐编辑",
            "套餐解读",
            "项目比对",
            "品牌选择",
            "品牌更换",
            "性能对比",
            "产品对比",
            "服务对比",
            "上架建议",
            "推荐",
            "对比",
            "选品",
            "品牌",
            "机滤选择",
        ),
        "推荐与对比",
        "推荐型号/项目、品牌或产品对比与选型（含上架/机滤等选型）。",
    ),
    (
        (
            "项目说明",
            "项目咨询",
            "项目确认",
            "项目加项",
            "项目关联",
            "项目区分",
            "项目梳理",
            "项目定义",
        ),
        "项目与内容",
        "保养项目含义、范围、加项/关联、区分梳理或内容确认。",
    ),
    (
        (
            "服务说明",
            "服务咨询",
            "服务范围",
            "套餐咨询",
            "套餐",
            "服务保障",
            "服务流程",
            "服务次数",
            "服务入口",
            "服务解释",
            "售后服务",
            "售后咨询",
        ),
        "服务与套餐",
        "服务条款、流程、次数、入口、保障、套餐内容、售后服务或范围相关。",
    ),
    (
        (
            "信息查询",
            "商品查询",
            "产品查询",
            "属性查询",
            "参数查询",
            "认证查询",
            "产地查询",
            "使用范围",
            "有效期",
            "类型判断",
            "品类区分",
            "名称释义",
            "真伪",
            "鉴别",
            "四季通用",
            "材质",
            "认证",
            "参数",
            "属性",
            "说明",
            "查询",
        ),
        "规格与信息",
        "查参数、属性、产地、认证、有效期、名称释义或说明类信息。",
    ),
    (
        ("单品购买", "配件更换", "配件", "单品", "购买"),
        "配件与购买",
        "配件/单品购买、更换或选购。",
    ),
    (
        ("咨询",),
        "通用咨询",
        "泛咨询类（未命中更具体主题时的兜底）。",
    ),
]

# 合并后仍单独展示的兜底说明（无关键词命中时）
SECONDARY_FALLBACK_MERGED = "其他（长尾细分）"
SECONDARY_FALLBACK_DESC = "模型细分标签较分散、未命中通用分组；可结合一级类目理解具体场景。"


def merge_secondary_category(raw_secondary: str) -> Tuple[str, str]:
    """
    将模型输出的二级类目合并为更粗的主题，并返回简短说明。
    返回：(合并后名称, 说明文案)
    """
    raw = ensure_nonempty_str(raw_secondary).strip()
    if not raw:
        raw = "其他"
    for needles, label, desc in SECONDARY_MERGE_GROUPS:
        if any(n in raw for n in needles):
            return label, desc
    return SECONDARY_FALLBACK_MERGED, SECONDARY_FALLBACK_DESC


def describe_merged_secondary(merged_label: str) -> str:
    """
    根据「合并后的二级类目名称」返回展示用说明（与 merge_secondary_category 的规则一致）。
    """
    label = ensure_nonempty_str(merged_label).strip()
    if label == SECONDARY_FALLBACK_MERGED:
        return SECONDARY_FALLBACK_DESC
    for _, grp_label, desc in SECONDARY_MERGE_GROUPS:
        if grp_label == label:
            return desc
    return SECONDARY_FALLBACK_DESC


def build_category_pair_table_html(map_results: List[Dict[str, Any]], top_n_pairs: Optional[int] = None) -> str:
    """
    你指定的两层查看方式：
    1) 一级类目独立看占比（占比基准=全量 Session）
    2) 每个一级类目内部看二级类目占比（占比基准=该一级类目内部）
    """
    total = max(1, len(map_results))
    primary_counter: Dict[str, int] = {}
    # secondary_counter[primary][merged_secondary] = {"total":..., "bad":...}
    secondary_counter: Dict[str, Dict[str, Dict[str, int]]] = {}

    # 不合理原因概要（基于 common_sense_reason 的关键词命中）
    # 注意：这里尽量不使用“内部规则/Prompt”相关词，只做用户体验/常识层面的归因概括。
    UNREASONABLE_REASON_KEYWORDS: List[Tuple[str, List[str]]] = [
        ("未响应关键诉求/失联", ["未响应", "不回应", "没有回应", "完全未响应", "失联", "回复为nan", "回答为nan"]),
        ("答非所问/不相关", ["答非所问", "不相关", "偏题", "没有解决", "无法解决用户的问题", "没解决", "不匹配"]),
        ("适配/规格不匹配", ["不适配", "不匹配", "适配错误", "型号不符", "规格不符", "不符合适配", "适配确认不清"]),
        ("常识错误", ["常识错误", "违反常识", "不符合常识", "常识问题"]),
        ("表达生硬/机械感", ["机械感", "生硬", "模板", "套话", "像模板"]),
        ("明显优化空间", ["明显优化", "有待优化", "待优化", "优化空间"]),
    ]
    for item in map_results:
        llm = item.get("llm_result") or {}
        primary = ensure_nonempty_str(llm.get("primary_category") or "其他")
        raw_secondary = ensure_nonempty_str(llm.get("secondary_category") or "其他")
        merged_secondary, _ = merge_secondary_category(raw_secondary)
        answer_issue = ensure_nonempty_str(llm.get("answer_issue"))
        common_sense_reason = ensure_nonempty_str(llm.get("common_sense_reason"))
        primary_counter[primary] = primary_counter.get(primary, 0) + 1
        secondary_counter.setdefault(primary, {})
        if merged_secondary not in secondary_counter[primary]:
            secondary_counter[primary][merged_secondary] = {"total": 0, "bad": 0}
        secondary_counter[primary][merged_secondary]["total"] += 1
        if answer_issue == "不合理":
            secondary_counter[primary][merged_secondary]["bad"] += 1
            # reason_keyword_counts 以“扩展字段”挂在 stats 上，避免改动外部结构太多
            stats = secondary_counter[primary][merged_secondary]
            if "reason_keyword_counts" not in stats:
                stats["reason_keyword_counts"] = {}  # type: ignore[assignment]
            if "reason_samples" not in stats:
                stats["reason_samples"] = []  # type: ignore[assignment]

            reason_text = common_sense_reason.strip()
            if reason_text:
                # 保存一个简短样本用于兜底
                stats["reason_samples"].append(truncate_text(reason_text, 70))  # type: ignore[index]
            for label, needles in UNREASONABLE_REASON_KEYWORDS:
                if any(n in reason_text for n in needles):
                    rc = stats["reason_keyword_counts"]  # type: ignore[index]
                    rc[label] = rc.get(label, 0) + 1  # type: ignore[index]

    def summarize_unreasonable_reasons(stats: Dict[str, Any]) -> str:
        keyword_counts: Dict[str, int] = stats.get("reason_keyword_counts") or {}
        samples: List[str] = stats.get("reason_samples") or []
        if keyword_counts:
            top = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            return "；".join([k for k, _ in top])
        if samples:
            # 兜底：用一个不合理样本的简短原因描述
            return samples[0]
        return "（原因文本为空，无法提炼）"

    primary_items = sorted(primary_counter.items(), key=lambda x: x[1], reverse=True)
    if top_n_pairs is not None:
        # top_n_pairs 在这里复用为“最多展示前 N 个一级类目”
        primary_items = primary_items[: max(1, top_n_pairs)]

    primary_rows_html: List[str] = []
    for primary, cnt in primary_items:
        share = cnt / total * 100
        primary_rows_html.append(
            "<tr>"
            f"<td style='word-break:break-all;padding:8px;'>{html_lib.escape(primary)}</td>"
            f"<td style='text-align:right;padding:8px;'>{cnt}</td>"
            f"<td style='text-align:right;padding:8px;'>{share:.2f}%</td>"
            "</tr>"
        )

    nested_blocks_html: List[str] = []
    for primary, primary_cnt in primary_items:
        sec_items = sorted(
            secondary_counter.get(primary, {}).items(),
            key=lambda x: x[1].get("total", 0),
            reverse=True,
        )
        sec_rows_html: List[str] = []
        for secondary, stats in sec_items:
            sec_cnt = stats.get("total", 0)
            sec_bad_cnt = stats.get("bad", 0)
            sec_share = sec_cnt / max(1, primary_cnt) * 100
            bad_share = sec_bad_cnt / max(1, sec_cnt) * 100
            sec_desc = describe_merged_secondary(secondary)
            bad_reason_summary = summarize_unreasonable_reasons(stats)
            sec_rows_html.append(
                "<tr>"
                f"<td style='word-break:break-all;padding:8px;'>{html_lib.escape(secondary)}</td>"
                f"<td style='word-break:break-all;padding:8px;color:#555;font-size:11px;line-height:1.45;'>{html_lib.escape(sec_desc)}</td>"
                f"<td style='text-align:right;padding:8px;'>{sec_cnt}</td>"
                f"<td style='text-align:right;padding:8px;'>{sec_share:.2f}%</td>"
                f"<td style='text-align:right;padding:8px;'>{bad_share:.2f}%</td>"
                f"<td style='word-break:break-all;padding:8px;color:#333;font-size:11px;line-height:1.45;'>{html_lib.escape(bad_reason_summary)}</td>"
                "</tr>"
            )

        sec_body_html = (
            "".join(sec_rows_html)
            if sec_rows_html
            else "<tr><td colspan='6'>暂无二级类目</td></tr>"
        )

        nested_blocks_html.append(
            "<details style='margin:10px 0;border:1px solid #eee;border-radius:8px;padding:8px 12px;'>"
            f"<summary style='cursor:pointer;color:#333;font-weight:600;'>{html_lib.escape(primary)}（数量 {primary_cnt}）</summary>"
            "<div style='max-height:380px;overflow:auto;margin-top:8px;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;'>"
            "<thead>"
            "<tr style='background:#fafafa;'>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:22%'>合并二级类目</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:44%'>说明</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:10%'>数量</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:14%'>占比（在该一级内）</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:10%'>不合理占比</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:30%'>不合理原因概要</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            f"{sec_body_html}"
            "</tbody>"
            "</table>"
            "</div>"
            "</details>"
        )

    return f"""
    <div style="padding:16px 24px;">
      <h3 style="margin:0 0 8px 0;">一级类目占比 + 二级类目占比（分层查看）</h3>
      <div style="color:#666;font-size:12px;margin-bottom:10px;">总 Session：{len(map_results)}（按 Map 阶段结果统计）。二级类目已按语义合并展示，占比按合并后口径统计。</div>

      <div style="max-height:520px;overflow:auto;border:1px solid #eee;border-radius:8px;margin-bottom:14px;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;">
          <thead>
            <tr style="background:#fafafa;">
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:55%;">一级类目</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:20%;">数量</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:25%;">占比（相对全量）</th>
            </tr>
          </thead>
          <tbody>
            {''.join(primary_rows_html) if primary_rows_html else "<tr><td colspan='3'>无数据</td></tr>"}
          </tbody>
        </table>
      </div>

      <div>
        {''.join(nested_blocks_html) if nested_blocks_html else "<div style='color:#666;font-size:12px;'>暂无二级类目数据</div>"}
      </div>
    </div>
    """


def detect_entrance_pagename_column(columns: List[str]) -> Optional[str]:
    """
    从原始输入表中识别 entrancepagename 列名。
    """
    for col in columns:
        if "entrancepagename" in str(col).lower():
            return col
    # 辅助兜底：包含 entrance + page
    for col in columns:
        lc = str(col).lower()
        if "entrance" in lc and "page" in lc and "name" in lc:
            return col
    return None


def build_session_entrance_map(
    input_path: str,
) -> Dict[Tuple[str, str], str]:
    """
    构建 (user_id, session_id) -> entrancepagename 映射。
    """
    df_raw = step1.read_input_file(input_path)
    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    if user_col is None:
        raise ValueError("无法识别 userid 字段，请检查表结构。")
    entrance_col = detect_entrance_pagename_column(columns)
    if entrance_col is None:
        raise ValueError("无法识别 entrancepagename 列，请检查表结构（列名可能包含 entrancepagename）。")

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    # 统一转字符串，避免分组时类型不一致
    df["_user_id"] = df[user_col].fillna("").astype(str).str.strip()
    df["_session_id"] = df[session_col].fillna("").astype(str).str.strip()
    df["_entrance"] = df[entrance_col].fillna("").astype(str).str.strip()

    # 若同一 session 内 entrance 不一致，取出现次数最多的那个
    entrance_map: Dict[Tuple[str, str], str] = {}
    for (u, s), g in df.groupby(["_user_id", "_session_id"]):
        vals = [v for v in g["_entrance"].tolist() if v]
        if not vals:
            entrance_map[(str(u), str(s))] = ""
        else:
            entrance_map[(str(u), str(s))] = Counter(vals).most_common(1)[0][0]
    return entrance_map


def build_category_pair_table_html_simple(map_results: List[Dict[str, Any]], total_title: str) -> str:
    """
    简化版“看数模块”：只展示一级类目占比 + 二级类目占比（不展示 answer_issue 相关列）。
    """
    total = max(1, len(map_results))
    primary_counter: Dict[str, int] = {}
    secondary_counter: Dict[str, Dict[str, int]] = {}

    for item in map_results:
        llm = item.get("llm_result") or {}
        primary = ensure_nonempty_str(llm.get("primary_category") or "其他")
        raw_secondary = ensure_nonempty_str(llm.get("secondary_category") or "其他")
        merged_secondary, _ = merge_secondary_category(raw_secondary)

        primary_counter[primary] = primary_counter.get(primary, 0) + 1
        secondary_counter.setdefault(primary, {})
        secondary_counter[primary][merged_secondary] = secondary_counter[primary].get(merged_secondary, 0) + 1

    primary_items = sorted(primary_counter.items(), key=lambda x: x[1], reverse=True)

    primary_rows_html: List[str] = []
    for primary, cnt in primary_items:
        share = cnt / total * 100
        primary_rows_html.append(
            "<tr>"
            f"<td style='word-break:break-all;padding:8px;'>{html_lib.escape(primary)}</td>"
            f"<td style='text-align:right;padding:8px;'>{cnt}</td>"
            f"<td style='text-align:right;padding:8px;'>{share:.2f}%</td>"
            "</tr>"
        )

    nested_blocks_html: List[str] = []
    for primary, primary_cnt in primary_items:
        sec_items = sorted(secondary_counter.get(primary, {}).items(), key=lambda x: x[1], reverse=True)
        sec_rows_html: List[str] = []
        for secondary, sec_cnt in sec_items:
            sec_share = sec_cnt / max(1, primary_cnt) * 100
            sec_desc = describe_merged_secondary(secondary)
            sec_rows_html.append(
                "<tr>"
                f"<td style='word-break:break-all;padding:8px;'>{html_lib.escape(secondary)}</td>"
                f"<td style='word-break:break-all;padding:8px;color:#555;font-size:11px;line-height:1.45;'>{html_lib.escape(sec_desc)}</td>"
                f"<td style='text-align:right;padding:8px;'>{sec_cnt}</td>"
                f"<td style='text-align:right;padding:8px;'>{sec_share:.2f}%</td>"
                "</tr>"
            )

        sec_body_html = "".join(sec_rows_html) if sec_rows_html else "<tr><td colspan='4'>暂无二级类目</td></tr>"
        nested_blocks_html.append(
            "<details style='margin:10px 0;border:1px solid #eee;border-radius:8px;padding:8px 12px;'>"
            f"<summary style='cursor:pointer;color:#333;font-weight:600;'>{html_lib.escape(primary)}（数量 {primary_cnt}）</summary>"
            "<div style='max-height:360px;overflow:auto;margin-top:8px;'>"
            "<table style='width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;'>"
            "<thead>"
            "<tr style='background:#fafafa;'>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:28%'>合并二级类目</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:44%'>说明</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:12%'>数量</th>"
            "<th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:16%'>占比（在该一级内）</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            f"{sec_body_html}"
            "</tbody>"
            "</table>"
            "</div>"
            "</details>"
        )

    return f"""
    <div style="padding:12px 0;">
      <div style="margin:0 0 10px 0;color:#222;">
        <h3 style="margin:0 0 6px 0;">看数模块：{html_lib.escape(total_title)}</h3>
        <div style="color:#666;font-size:12px;">该入口下 Session 总数：{len(map_results)}；一级类目占比基于该入口 Session 总数。</div>
      </div>

      <div style="max-height:520px;overflow:auto;border:1px solid #eee;border-radius:8px;margin-bottom:14px;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;">
          <thead>
            <tr style="background:#fafafa;">
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:55%;">一级类目</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:20%;">数量</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:25%;">占比（相对该入口全量）</th>
            </tr>
          </thead>
          <tbody>
            {''.join(primary_rows_html) if primary_rows_html else "<tr><td colspan='3'>无数据</td></tr>"}
          </tbody>
        </table>
      </div>

      <div>
        {''.join(nested_blocks_html) if nested_blocks_html else "<div style='color:#666;font-size:12px;'>暂无二级类目数据</div>"}
      </div>
    </div>
    """


def build_entrance_modules_html(
    map_results: List[Dict[str, Any]],
    input_path: str,
    entrance_names: List[str],
) -> str:
    """
    按 entrancepagename 生成“看数模块”：一级类目占比 + 二级类目占比。
    """
    try:
        session_entrance_map = build_session_entrance_map(input_path)
    except Exception as exc:
        return f"""
        <div style="padding:16px 24px;">
          <h3 style="margin:0 0 8px 0;">看数模块（入口维度）</h3>
          <div style="color:#b00;font-size:13px;">入口模块生成失败：{html_lib.escape(str(exc))}</div>
        </div>
        """

    modules: List[str] = []
    for entrance_name in entrance_names:
        subset: List[Dict[str, Any]] = []
        for item in map_results:
            key = (str(item.get("user_id") or "").strip(), str(item.get("session_id") or "").strip())
            if session_entrance_map.get(key) == entrance_name:
                subset.append(item)

        modules.append(build_category_pair_table_html_simple(subset, total_title=entrance_name))

    return "".join(modules)


def load_intent_key_to_zh_desc(mapping_xlsx_path: str) -> Dict[str, str]:
    """
    从「意图描述和对应专家策略.xlsx」读取「意图」->「意图描述」映射。
    同一意图多行时，保留较长的描述文本（信息更完整）。
    """
    if not os.path.exists(mapping_xlsx_path):
        return {}
    try:
        df = pd.read_excel(mapping_xlsx_path)
    except Exception:
        return {}
    intent_col = None
    desc_col = None
    for c in df.columns:
        s = str(c).strip()
        if s == "意图":
            intent_col = c
        if s == "意图描述":
            desc_col = c
    if intent_col is None or desc_col is None:
        return {}

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        key = str(row.get(intent_col, "") or "").strip()
        desc = str(row.get(desc_col, "") or "").strip()
        if not key:
            continue
        if key not in out or len(desc) > len(out.get(key, "")):
            out[key] = desc
    return out


def build_intent_unreasonable_table_html(
    map_results: List[Dict[str, Any]],
    mapping_xlsx_path: str,
    min_sessions_main: int = 5,
) -> str:
    """
    意图维度：统计各 mapping_intent_desc（与映射表「意图」列一致）下 answer_issue=不合理 的占比。
    """
    zh_map = load_intent_key_to_zh_desc(mapping_xlsx_path)

    stats: Dict[str, Dict[str, int]] = {}
    for item in map_results:
        intent_key = ensure_nonempty_str(item.get("mapping_intent_desc") or "未知")
        llm = item.get("llm_result") or {}
        ai = ensure_nonempty_str(llm.get("answer_issue"))
        if intent_key not in stats:
            stats[intent_key] = {"total": 0, "bad": 0}
        stats[intent_key]["total"] += 1
        if ai == "不合理":
            stats[intent_key]["bad"] += 1

    def row_tuple(k: str) -> Tuple[str, int, int, float]:
        st = stats[k]
        t, b = st["total"], st["bad"]
        return (k, t, b, b / max(1, t) * 100)

    all_keys = list(stats.keys())
    main_rows = [row_tuple(k) for k in all_keys if stats[k]["total"] >= min_sessions_main]
    main_rows.sort(key=lambda x: (x[3], x[1]), reverse=True)

    small_rows = [row_tuple(k) for k in all_keys if stats[k]["total"] < min_sessions_main]
    small_rows.sort(key=lambda x: (x[3], x[1]), reverse=True)

    def render_rows(rows: List[Tuple[str, int, int, float]]) -> str:
        parts: List[str] = []
        for k, t, b, pct in rows:
            zh = zh_map.get(k, "")
            zh_short = truncate_text(zh, 140) if zh else "（映射表中未找到对应「意图描述」，以意图 key 为准）"
            parts.append(
                "<tr>"
                f"<td style='word-break:break-all;padding:8px;'>{html_lib.escape(k)}</td>"
                f"<td style='word-break:break-all;padding:8px;color:#555;font-size:11px;line-height:1.45;'>{html_lib.escape(zh_short)}</td>"
                f"<td style='text-align:right;padding:8px;'>{t}</td>"
                f"<td style='text-align:right;padding:8px;'>{b}</td>"
                f"<td style='text-align:right;padding:8px;'>{pct:.2f}%</td>"
                "</tr>"
            )
        return "".join(parts) if parts else "<tr><td colspan='5'>无数据</td></tr>"

    main_body = render_rows(main_rows)
    small_body = render_rows(small_rows[:15])

    return f"""
    <div style="padding:16px 24px;">
      <h3 style="margin:0 0 8px 0;">意图维度：不合理占比（answer_issue=不合理）</h3>
      <div style="color:#666;font-size:12px;margin-bottom:10px;">
        统计口径：按 Map 结果中的 <code>mapping_intent_desc</code>（与「意图描述和对应专家策略.xlsx」的「意图」列对齐）聚合；
        「不合理占比」= 该意图下不合理会话数 / 该意图总会话数。
        主表仅展示会话数 ≥ {min_sessions_main} 的意图（避免小样本误判）；下方附小样本参考（最多 15 行）。
      </div>

      <div style="max-height:420px;overflow:auto;border:1px solid #eee;border-radius:8px;margin-bottom:14px;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;">
          <thead>
            <tr style="background:#fafafa;">
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:16%;">意图（key）</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:44%;">意图说明（映射表）</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:10%;">会话数</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:12%;">不合理数</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:right;width:18%;">不合理占比</th>
            </tr>
          </thead>
          <tbody>
            {main_body}
          </tbody>
        </table>
      </div>

      <details style="border:1px solid #eee;border-radius:8px;padding:8px 12px;">
        <summary style="cursor:pointer;color:#333;font-weight:600;">小样本意图参考（会话数 &lt; {min_sessions_main}，最多展示 15 条）</summary>
        <div style="max-height:280px;overflow:auto;margin-top:8px;">
          <table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;">
            <thead>
              <tr style="background:#fafafa;">
                <th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:16%;'>意图（key）</th>
                <th style='border-bottom:1px solid #eee;padding:8px;text-align:left;width:44%;'>意图说明（映射表）</th>
                <th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:10%;'>会话数</th>
                <th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:12%;'>不合理数</th>
                <th style='border-bottom:1px solid #eee;padding:8px;text-align:right;width:18%;'>不合理占比</th>
              </tr>
            </thead>
            <tbody>
              {small_body}
            </tbody>
          </table>
        </div>
      </details>
    </div>
    """


def build_typical_cases_table_html(map_results: List[Dict[str, Any]], limit: Optional[int] = None) -> str:
    """
    替代 pyecharts Table：展示不全问题改为纯 HTML + 可折叠查看完整内容。
    """
    typical: List[Dict[str, Any]] = []
    for item in map_results:
        llm = item.get("llm_result") or {}
        if llm.get("is_typical") is True and llm.get("prompt_defect_suspected") is True:
            typical.append(
                {
                    "user_id": item.get("user_id"),
                    "session_id": item.get("session_id"),
                    "primary_category": llm.get("primary_category"),
                    "secondary_category": llm.get("secondary_category"),
                    "common_sense_reason": llm.get("common_sense_reason"),
                    "defect_reason": llm.get("defect_reason"),
                }
            )

    if limit is not None:
        typical = typical[: max(1, limit)]

    rows_html: List[str] = []
    for t in typical:
        user_id = html_lib.escape(ensure_nonempty_str(t.get("user_id")))
        session_id = html_lib.escape(str(t.get("session_id") or ""))
        primary = html_lib.escape(ensure_nonempty_str(t.get("primary_category")))
        secondary = html_lib.escape(ensure_nonempty_str(t.get("secondary_category")))
        cs_reason_full = ensure_nonempty_str(t.get("common_sense_reason"))
        defect_reason_full = ensure_nonempty_str(t.get("defect_reason"))
        cs_short = truncate_text(cs_reason_full, 160)
        defect_short = truncate_text(defect_reason_full, 160)

        rows_html.append(
            "<tr>"
            f"<td style='word-break:break-all;padding:8px;'>{user_id}</td>"
            f"<td style='word-break:break-all;padding:8px;'>{session_id}</td>"
            f"<td style='word-break:break-all;padding:8px;'>{primary}</td>"
            f"<td style='word-break:break-all;padding:8px;'>{secondary}</td>"
            "<td style='padding:8px;max-width:360px;white-space:normal;'>"
            f"<div style='color:#333;'>{html_lib.escape(cs_short)}</div>"
            f"<details><summary style='cursor:pointer;color:#2f6feb;'>查看完整</summary>"
            f"<pre style='margin:8px 0 0 0;white-space:pre-wrap;word-break:break-word;'>{html_lib.escape(cs_reason_full)}</pre>"
            "</details>"
            "</td>"
            "<td style='padding:8px;max-width:420px;white-space:normal;'>"
            f"<div style='color:#333;'>{html_lib.escape(defect_short)}</div>"
            f"<details><summary style='cursor:pointer;color:#2f6feb;'>查看完整</summary>"
            f"<pre style='margin:8px 0 0 0;white-space:pre-wrap;word-break:break-word;'>{html_lib.escape(defect_reason_full)}</pre>"
            "</details>"
            "</td>"
            "</tr>"
        )

    return f"""
    <div style="padding:16px 24px;">
      <h3 style="margin:0 0 8px 0;">典型坏案列表（is_typical + prompt_defect_suspected）</h3>
      <div style="color:#666;font-size:12px;margin-bottom:10px;">共 {len(typical)} 条；点击“查看完整”可展开 common_sense_reason / defect_reason。</div>
      <div style="max-height:780px;overflow:auto;border:1px solid #eee;border-radius:8px;">
        <table style="width:100%;border-collapse:collapse;font-size:12px;table-layout:fixed;">
          <thead>
            <tr style="background:#fafafa;">
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:14%;">UserID</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:12%;">SessionID</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:12%;">一级类目</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:12%;">二级类目</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:20%;">常识体验原因</th>
              <th style="position:sticky;top:0;z-index:1;border-bottom:1px solid #eee;padding:8px;text-align:left;width:30%;">规则疑点/缺陷原因</th>
            </tr>
          </thead>
          <tbody>
            {''.join(rows_html) if rows_html else "<tr><td colspan='6'>暂无典型坏案</td></tr>"}
          </tbody>
        </table>
      </div>
    </div>
    """


def compute_health_stats_by_category_pair(map_results: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    计算每个「一级类目-二级类目」的健康度（answer_issue 维度）：
    - 合理占比 = answer_issue == "合理" 的比例
    - 不合理占比 = answer_issue == "不合理" 的比例
    二级类目口径使用 merge_secondary_category（与报告展示保持一致）。
    """
    stats: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for item in map_results:
        llm = item.get("llm_result") or {}
        primary = ensure_nonempty_str(llm.get("primary_category") or "其他").strip()
        raw_secondary = ensure_nonempty_str(llm.get("secondary_category") or "其他").strip()
        merged_secondary, _ = merge_secondary_category(raw_secondary)
        answer_issue = ensure_nonempty_str(llm.get("answer_issue")).strip()
        rewrite_issue = ensure_nonempty_str(llm.get("rewrite_issue")).strip()
        prompt_defect_suspected = llm.get("prompt_defect_suspected") is True

        key = (primary, merged_secondary)
        if key not in stats:
            stats[key] = {
                "primary": primary,
                "secondary": merged_secondary,
                "total": 0,
                "good": 0,
                "bad": 0,
                "rewrite_good": 0,
                "prompt_defect_true": 0,
            }

        stats[key]["total"] += 1
        if answer_issue == "合理":
            stats[key]["good"] += 1
        elif answer_issue == "不合理":
            stats[key]["bad"] += 1

        if rewrite_issue == "合理":
            stats[key]["rewrite_good"] += 1
        if prompt_defect_suspected:
            stats[key]["prompt_defect_true"] += 1

    return stats


def format_pair_label(primary: str, secondary: str) -> str:
    return f"{primary} - {secondary}"


def build_red_black_board_html(
    pair_stats: Dict[Tuple[str, str], Dict[str, Any]],
    red_topk: int = 3,
    black_topk: int = 5,
    min_total: int = 5,
) -> Tuple[str, List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    构建【大盘红黑榜】：
    - 红榜：合理占比最高 top3
    - 黑榜：不合理占比最高 top3~5
    """
    overall_total = sum(int(v.get("total", 0)) for v in pair_stats.values()) or 1
    eligible = [k for k, v in pair_stats.items() if v.get("total", 0) >= min_total]
    if not eligible:
        eligible = list(pair_stats.keys())

    red_pairs = sorted(
        eligible,
        # 红榜：按“合理绝对数量”降序（不看合理率），避免低流量长尾误导
        key=lambda k: pair_stats[k].get("good", 0),
        reverse=True,
    )[: max(1, red_topk)]

    black_pairs = sorted(
        eligible,
        key=lambda k: pair_stats[k].get("bad", 0),
        reverse=True,
    )[: max(1, black_topk)]

    def one_sentence_for_pair(k: Tuple[str, str], is_red: bool) -> str:
        v = pair_stats[k]
        total = max(1, v.get("total", 1))
        good_pct = v.get("good", 0) / total * 100
        bad_pct = v.get("bad", 0) / total * 100
        rewrite_good_pct = v.get("rewrite_good", 0) / total * 100
        prompt_defect_pct = v.get("prompt_defect_true", 0) / total * 100

        if is_red:
            return f"合理占比高（{good_pct:.1f}%），且 rewrite_issue 也更常为合理（{rewrite_good_pct:.1f}%），提示整体引导/回答质量更稳定。"
        return f"不合理占比高（{bad_pct:.1f}%），且 prompt 缺陷疑似比例偏高（{prompt_defect_pct:.1f}%）或 rewrite_issue 质量不足（合理仅 {rewrite_good_pct:.1f}%），值得优先排查策略/流程缺口。"

    red_rows: List[str] = []
    for k in red_pairs:
        v = pair_stats[k]
        total = max(1, v.get("total", 1))
        good_pct = v.get("good", 0) / total * 100
        good_cnt = int(v.get("good", 0))
        label = format_pair_label(v["primary"], v["secondary"])
        red_rows.append(
            "<div style='border:1px solid #ffb3b3;border-radius:10px;padding:10px 12px;background:#fff5f5;margin:8px 0;'>"
            f"<div style='font-weight:700;color:#b60000;'>{html_lib.escape(label)}（合理占比 {good_pct:.1f}% | 合理 Case 数: {good_cnt}条）</div>"
            f"<div style='color:#444;font-size:12px;margin-top:6px;'>{html_lib.escape(one_sentence_for_pair(k, is_red=True))}</div>"
            "</div>"
        )

    black_rows: List[str] = []
    for k in black_pairs:
        v = pair_stats[k]
        total = max(1, v.get("total", 1))
        bad_cnt = int(v.get("bad", 0))
        bad_pct = bad_cnt / total * 100
        flow_pct = total / overall_total * 100
        label = format_pair_label(v["primary"], v["secondary"])
        black_rows.append(
            "<div style='border:1px solid #222;border-radius:10px;padding:10px 12px;background:#f7f7f7;margin:8px 0;'>"
            f"<div style='font-weight:700;color:#111;'>{html_lib.escape(label)}（大盘流量占比: {flow_pct:.1f}% | 不合理占比: {bad_pct:.0f}% | 不良 Case 数: {bad_cnt}条）</div>"
            f"<div style='color:#444;font-size:12px;margin-top:6px;'>{html_lib.escape(one_sentence_for_pair(k, is_red=False))}</div>"
            "</div>"
        )

    board_html = f"""
    <div style="padding:10px 24px 0 24px;">
      <div style="display:flex;gap:18px;flex-wrap:wrap;">
        <div style="flex:1;min-width:320px;">
          <div style="font-size:14px;font-weight:800;color:#b60000;margin:8px 0;">大盘红榜（表现优异）</div>
          {''.join(red_rows) if red_rows else "<div style='color:#666;font-size:12px;'>无数据</div>"}
        </div>
        <div style="flex:1;min-width:320px;">
          <div style="font-size:14px;font-weight:800;color:#111;margin:8px 0;">大盘黑榜（亟待优化重灾区）</div>
          {''.join(black_rows) if black_rows else "<div style='color:#666;font-size:12px;'>无数据</div>"}
        </div>
      </div>
    </div>
    """
    return board_html, red_pairs, black_pairs


def build_primary_top5_black_modules_html(
    pair_stats: Dict[Tuple[str, str], Dict[str, Any]],
    top_primary: int = 5,
    black_topk_per_primary: int = 3,
    min_total: int = 5,
) -> str:
    """
    除了大盘黑榜，再增加“前5个一级类目下，每个一级类目内部的黑榜”模块。
    黑榜排序：按 bad（不合理绝对数量）降序。
    """
    overall_total = sum(int(v.get("total", 0)) for v in pair_stats.values()) or 1
    primary_total: Dict[str, int] = {}
    primary_bad: Dict[str, int] = {}
    for (primary, _secondary), v in pair_stats.items():
        t = int(v.get("total", 0))
        b = int(v.get("bad", 0))
        primary_total[primary] = primary_total.get(primary, 0) + t
        primary_bad[primary] = primary_bad.get(primary, 0) + b

    primary_items = sorted(primary_total.items(), key=lambda x: x[1], reverse=True)[:top_primary]
    blocks: List[str] = []

    blocks.append(
        "<div style=\"padding:10px 24px 0 24px;\">"
        "<h3 style=\"margin:0 0 8px 0;\">前5个一级类目下的黑榜（类目内部按不合理绝对数量排序）</h3>"
        f"<div style=\"color:#666;font-size:12px;margin-bottom:10px;\">大盘总 Session：{overall_total}</div>"
    )

    for primary, p_total in primary_items:
        p_flow_pct = p_total / overall_total * 100
        p_bad = primary_bad.get(primary, 0)

        # 该一级下所有二级组合
        pair_list: List[Tuple[str, Dict[str, Any]]] = []
        for (p, s), v in pair_stats.items():
            if p != primary:
                continue
            pair_list.append((s, v))

        eligible = [(s, v) for (s, v) in pair_list if int(v.get("total", 0)) >= min_total]
        if not eligible:
            eligible = pair_list

        eligible.sort(key=lambda x: int(x[1].get("bad", 0)), reverse=True)
        top_pairs = eligible[: max(1, black_topk_per_primary)]

        inner_rows: List[str] = []
        for secondary, v in top_pairs:
            sec_total = max(1, int(v.get("total", 0)))
            sec_bad = int(v.get("bad", 0))
            sec_bad_pct = sec_bad / sec_total * 100
            sec_flow_pct = sec_total / overall_total * 100
            label = format_pair_label(primary, secondary)
            inner_rows.append(
                "<div style='border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:8px 0;background:#fafafa;'>"
                f"<div style='font-weight:700;'>{html_lib.escape(label)}（大盘流量占比: {sec_flow_pct:.1f}% | 不合理占比: {sec_bad_pct:.0f}% | 不良 Case 数: {sec_bad}条）</div>"
                "</div>"
            )

        inner_content = (
            "".join(inner_rows)
            if inner_rows
            else "<div style='color:#666;font-size:12px;'>暂无二级数据</div>"
        )
        blocks.append(
            "<details style='border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:10px 0;background:#fff;'>"
            f"<summary style='cursor:pointer;color:#111;font-weight:700;'>{html_lib.escape(primary)}（一级流量占比: {p_flow_pct:.1f}% | 不合理总量: {p_bad}条）</summary>"
            f"{inner_content}"
            "</details>"
        )

    blocks.append("</div>")
    return "".join(blocks)


def extract_user_and_bot_from_session_history(session_history_text: str, max_turns: int = 2) -> Tuple[str, str]:
    """
    从 session_history_text 中提取用户原话与客服原话（提取前 N 轮里的“用户问题/客服回答”）。
    """
    text = ensure_nonempty_str(session_history_text)
    user_lines: List[str] = []
    bot_lines: List[str] = []
    pending_user: List[str] = []
    pending_bot: List[str] = []
    turns_seen = 0
    collect_target: Optional[str] = None  # "user" / "bot" / None

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("[轮次"):
            if pending_user or pending_bot:
                user_lines.extend(pending_user)
                bot_lines.extend(pending_bot)
                pending_user = []
                pending_bot = []
                turns_seen += 1
                if turns_seen >= max_turns:
                    break
            continue
        # 新老两种模板都兼容：
        # - 旧：用户问题：... / 客服回答：...
        # - 新：用户问题：... / 客服回答（已合并分段）：...
        if line.startswith("用户问题："):
            pending_user.append(line.replace("用户问题：", "", 1).strip())
            collect_target = "user"
            continue
        if line.startswith("客服回答：") or line.startswith("客服回答（已合并分段）："):
            if line.startswith("客服回答（已合并分段）："):
                bot_val = line.replace("客服回答（已合并分段）：", "", 1).strip()
            else:
                bot_val = line.replace("客服回答：", "", 1).strip()
            pending_bot.append(bot_val)
            collect_target = "bot"
            continue

        # 处理“拼接回答换行”续行：例如
        # 客服回答（已合并分段）：第一句
        # 第二句
        # 第三句
        # 这些续行应继续并入客服原话
        if line.startswith("重写问题：") or line.startswith("【rewrite_history"):
            collect_target = None
            continue
        if line.startswith("【") and line.endswith("】"):
            collect_target = None
            continue
        if collect_target == "bot":
            pending_bot.append(line)

    if turns_seen < max_turns and (pending_user or pending_bot):
        user_lines.extend(pending_user)
        bot_lines.extend(pending_bot)

    return "\n".join(user_lines).strip(), "\n".join(bot_lines).strip()


def pick_typical_bad_cases(bad_items: List[Dict[str, Any]], topk: int = 3) -> List[Dict[str, Any]]:
    """
    选择典型 bad case：优先 is_typical=True，其次按 common_sense_reason 长度排序。
    """
    def score(it: Dict[str, Any]) -> Tuple[int, int]:
        llm = it.get("llm_result") or {}
        is_typical = 1 if llm.get("is_typical") is True else 0
        reason = ensure_nonempty_str(llm.get("common_sense_reason"))
        return (is_typical, len(reason))

    sorted_items = sorted(bad_items, key=score, reverse=True)
    return sorted_items[: max(1, topk)]


def call_llm_for_topic_clustering(
    client: OpenAI,
    model_name: str,
    category_label: str,
    cases_text: str,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """
    黑榜类目专项诊断：主题聚类分析（只输出 JSON）。
    """
    system_prompt = "你是一个高级数据分析师。你只能基于输入数据进行聚类与归因推断。"
    user_prompt = (
        f"你是一个高级数据分析师。以下是【{category_label}】下所有被判定为不合理的客服对话记录和初步原因。"
        "请帮我进行『主题聚类分析』，输出以下内容："
        "1. 该类目下最核心的 1-2 个【普遍性/系统性缺陷】是什么？（用中文描述，并尽量具体到“场景+表现+后果”）"
        "2. 这种缺陷是系统流程问题，还是个别用户的特殊提问？请给出结论。\n\n"
        "输出要求：只输出一个 JSON 对象，不要输出多余文本。JSON 字段：\n"
        '{ "core_defects": ["缺陷1","缺陷2"], "systemic_vs_case": "系统流程问题|个别用户特殊提问|两者都有", "conclusion": "一句话总结" }\n\n'
        f"被判定不合理的样本如下（包含原始对话 + common_sense_reason 初步原因）：\n{cases_text}"
    )

    resp_text = call_llm_for_map(
        client=client,
        model_name=model_name,
        system_prompt=system_prompt,
        user_content=user_prompt,
        max_retries=max_retries,
    )
    try:
        cleaned = sanitize_llm_output_as_json(resp_text)
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    return {
        "core_defects": [],
        "systemic_vs_case": "系统流程问题",
        "conclusion": "主题聚类解析失败/返回格式不符合要求，建议人工复核。",
        "raw": resp_text[:1000],
    }


def build_black_pairs_topic_diagnosis(
    map_results: List[Dict[str, Any]],
    black_pairs: List[Tuple[str, str]],
    topic_cache_path: str,
    force_topic_diagnosis: bool = False,
    max_topic_bad_cases_chars: int = 25000,
) -> Dict[str, Any]:
    """
    针对每个黑榜类目做 1 次 LLM 专项诊断（带缓存）。
    """
    if os.path.exists(topic_cache_path) and not force_topic_diagnosis:
        try:
            with open(topic_cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            if isinstance(cache, dict):
                return cache
        except Exception:
            pass

    client = OpenAI(api_key=step2.API_KEY, base_url=step2.API_BASE_URL)
    model_name = step2.MODEL_NAME

    diagnosis_map: Dict[str, Any] = {}
    for primary, secondary in black_pairs:
        label = format_pair_label(primary, secondary)

        bad_items: List[Dict[str, Any]] = []
        for it in map_results:
            llm = it.get("llm_result") or {}
            p = ensure_nonempty_str(llm.get("primary_category") or "其他").strip()
            raw_sec = ensure_nonempty_str(llm.get("secondary_category") or "其他").strip()
            msec, _ = merge_secondary_category(raw_sec)
            ai = ensure_nonempty_str(llm.get("answer_issue")).strip()
            if p == primary and msec == secondary and ai == "不合理":
                bad_items.append(it)

        cases_blocks: List[str] = []
        used_chars = 0
        per_case_limit = 2500
        for idx, it in enumerate(bad_items[:200], start=1):
            llm = it.get("llm_result") or {}
            reason = ensure_nonempty_str(llm.get("common_sense_reason"))
            session_text = ensure_nonempty_str(it.get("session_history_text"))
            reason_short = reason[:500]
            session_short = session_text[:per_case_limit]
            block = (
                f"[案例 {idx}] SessionID={it.get('session_id')} UserID={it.get('user_id')}\n"
                f"客服对话原文：\n{session_short}\n"
                f"common_sense_reason 初步原因：{reason_short}\n"
            )
            if used_chars + len(block) > max_topic_bad_cases_chars:
                break
            cases_blocks.append(block)
            used_chars += len(block)

        cases_text = "\n\n".join(cases_blocks) if cases_blocks else "（无样本）"
        diagnosis = call_llm_for_topic_clustering(
            client=client,
            model_name=model_name,
            category_label=label,
            cases_text=cases_text,
            max_retries=3,
        )
        diagnosis["bad_items_count"] = len(bad_items)
        diagnosis["used_cases_chars"] = used_chars
        diagnosis_map[label] = diagnosis

    try:
        with open(topic_cache_path, "w", encoding="utf-8") as f:
            json.dump(diagnosis_map, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"警告：主题聚类缓存写入失败：{exc}")

    return diagnosis_map


def build_black_pairs_details_html(
    map_results: List[Dict[str, Any]],
    pair_stats: Dict[Tuple[str, str], Dict[str, Any]],
    black_pairs: List[Tuple[str, str]],
    diagnosis_map: Dict[str, Any],
) -> str:
    """
    将每个黑榜类目的诊断结论与典型 bad case 渲染到 HTML。
    """
    blocks: List[str] = []
    for primary, secondary in black_pairs:
        key = (primary, secondary)
        v = pair_stats.get(key) or {"total": 0, "bad": 0}
        total = max(1, v.get("total", 1))
        bad_pct = v.get("bad", 0) / total * 100
        label = format_pair_label(primary, secondary)

        diagnosis = diagnosis_map.get(label) or {}
        core_defects = diagnosis.get("core_defects") or []
        systemic_vs_case = ensure_nonempty_str(diagnosis.get("systemic_vs_case") or "系统流程问题")
        conclusion = ensure_nonempty_str(diagnosis.get("conclusion"))
        if not conclusion:
            conclusion = ""

        bad_items: List[Dict[str, Any]] = []
        for it in map_results:
            llm = it.get("llm_result") or {}
            p = ensure_nonempty_str(llm.get("primary_category") or "其他").strip()
            raw_sec = ensure_nonempty_str(llm.get("secondary_category") or "其他").strip()
            msec, _ = merge_secondary_category(raw_sec)
            ai = ensure_nonempty_str(llm.get("answer_issue")).strip()
            if p == primary and msec == secondary and ai == "不合理":
                bad_items.append(it)
        typical_bad_items = pick_typical_bad_cases(bad_items, topk=3)

        # 如果 LLM 缓存缺失/调用失败导致结论为空，使用本地启发式生成兜底结论
        fallback_need = (not core_defects) or (not conclusion) or ("暂无AI聚类诊断内容" in conclusion)
        if fallback_need:
            # 基于 common_sense_reason 做关键词归纳（只做体验/常识层面的归因概括）
            UNREASONABLE_REASON_KEYWORDS: List[Tuple[str, List[str]]] = [
                ("未响应/失联", ["未响应", "不回应", "没有回应", "完全未响应", "失联", "回复为nan", "回答为nan"]),
                ("答非所问/不相关", ["答非所问", "不相关", "偏题", "没有解决", "无法解决", "没解决", "不匹配"]),
                ("适配/规格不匹配", ["不适配", "不匹配", "适配错误", "型号不符", "规格不符", "不符合适配", "适配确认不清"]),
                ("常识错误", ["常识错误", "违反常识", "不符合常识", "常识问题"]),
                ("机械/模板式回复", ["机械感", "生硬", "模板", "套话", "像模板", "敷衍", "回避"]),
                ("承接不到位/无法决策", ["承接", "未承接", "无法承接", "对比未承接", "场景未承接", "无法解决用户", "陷入死循环"]),
            ]
            keyword_counts: Dict[str, int] = {}
            prompt_defect_true_cnt = 0
            for it in bad_items:
                llm = it.get("llm_result") or {}
                if llm.get("prompt_defect_suspected") is True:
                    prompt_defect_true_cnt += 1
                reason_text = ensure_nonempty_str(llm.get("common_sense_reason"))
                for label_kw, needles in UNREASONABLE_REASON_KEYWORDS:
                    if any(n in reason_text for n in needles):
                        keyword_counts[label_kw] = keyword_counts.get(label_kw, 0) + 1

            top_kw = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)[:2]
            if top_kw:
                kw_labels = [x[0] for x in top_kw if x[1] > 0]
                if not kw_labels:
                    kw_labels = [top_kw[0][0]]
            else:
                kw_labels = ["意图承接不足/无法解决核心疑虑"]

            prompt_defect_ratio = prompt_defect_true_cnt / max(1, len(bad_items))
            # 不强行判断“个别还是系统”，但默认给“系统流程问题”的更利于产品迭代
            systemic_vs_case = "系统流程问题" if prompt_defect_ratio >= 0.25 else "两者都有"
            core_defects = [kw_labels[0]]
            if len(kw_labels) >= 2:
                core_defects.append(kw_labels[1])

            conclusion = f"该类目不合理主要集中在：{ ' / '.join(kw_labels[:2]) }，导致用户关键疑虑无法被持续承接并推进决策，体验阻塞明显。"

        defect_html = ""
        if core_defects:
            defect_html = "<div style='margin-top:6px;color:#111;font-size:13px;line-height:1.6;'>" + "".join(
                [f"<div>• {html_lib.escape(ensure_nonempty_str(d))}</div>" for d in core_defects[:2]]
            ) + "</div>"

        cards: List[str] = []
        for i, it in enumerate(typical_bad_items, start=1):
            llm = it.get("llm_result") or {}
            reason = ensure_nonempty_str(llm.get("common_sense_reason"))
            session_text = ensure_nonempty_str(it.get("session_history_text"))
            user_origin, bot_origin = extract_user_and_bot_from_session_history(session_text, max_turns=2)
            cards.append(
                "<div style='border:1px solid #eee;border-radius:10px;padding:10px 12px;margin:10px 0;'>"
                f"<div style='font-weight:700;'>典型案例 {i}（SessionID={html_lib.escape(str(it.get('session_id') or ''))}）</div>"
                "<div style='margin-top:8px;'>"
                "<div style='font-weight:600;font-size:12px;color:#333;'>用户原话</div>"
                f"<pre style='background:#fafafa;padding:10px;border-radius:8px;white-space:pre-wrap;word-break:break-word;'>{html_lib.escape(user_origin) or '（未解析到用户原话）'}</pre>"
                "<div style='font-weight:600;font-size:12px;color:#333;margin-top:8px;'>客服原话</div>"
                f"<pre style='background:#fafafa;padding:10px;border-radius:8px;white-space:pre-wrap;word-break:break-word;'>{html_lib.escape(bot_origin) or '（未解析到客服原话）'}</pre>"
                "<div style='font-weight:600;font-size:12px;color:#333;margin-top:8px;'>评价（common_sense_reason）</div>"
                f"<div style='margin-top:4px;color:#111;font-size:12px;line-height:1.6;'>{html_lib.escape(reason)}</div>"
                "</div>"
                "</div>"
            )

        cards_html = (
            "".join(cards)
            if cards
            else "<div style='color:#666;font-size:12px;'>暂无典型样本</div>"
        )

        # 列出该类目下所有不合理 SessionID，供产品定位
        bad_session_ids: List[str] = []
        for it in bad_items:
            sid = ensure_nonempty_str(it.get("session_id"))
            if sid:
                bad_session_ids.append(sid)
        bad_session_ids = sorted(list(dict.fromkeys(bad_session_ids)))
        bad_session_block = html_lib.escape("，".join(bad_session_ids)) if bad_session_ids else "（无 SessionID）"

        blocks.append(
            "<div style='padding:16px 24px;'>"
            f"<h3 style='margin:0 0 8px 0;'>{html_lib.escape(label)}（不合理率 {bad_pct:.1f}%）</h3>"
            "<div style='background:#fff3cd;border:1px solid #ffeeba;border-radius:10px;padding:12px 14px;'>"
            "<div style='font-weight:900;'>⚠️ 核心诊断结论</div>"
            f"<div style='margin-top:6px;color:#111;line-height:1.6;font-size:13px;'>{html_lib.escape(systemic_vs_case)}：{html_lib.escape(conclusion)}</div>"
            f"{defect_html}"
            "</div>"
            "<div style='font-weight:900;margin-top:10px;'>🔍 典型案例支撑</div>"
            f"{cards_html}"
            "<div style='font-weight:900;margin-top:10px;'>不合理 SessionID 列表</div>"
            f"<div style='max-height:180px;overflow:auto;border:1px solid #eee;border-radius:8px;padding:10px 12px;font-family:monospace;font-size:12px;white-space:pre-wrap;word-break:break-word;margin-top:6px;color:#333;'>{bad_session_block}</div>"
            "</div>"
        )

    return "".join(blocks)


def _looks_like_url_or_media(text: str) -> bool:
    """
    过滤掉“纯链接/图片”等用户原话噪声，避免影响“用户真实问题清单”质量。
    """
    t = (text or "").strip()
    if not t:
        return True
    if t.startswith("http://") or t.startswith("https://"):
        return True
    if t.startswith("tuhu://"):
        return True
    # 典型图片链接路径（有时用户问题会被提取成图片URL）
    if "tigertalk/im/" in t:
        return True
    if re.search(r"\.(png|jpg|jpeg|gif|webp)(\?.*)?$", t, flags=re.IGNORECASE):
        return True
    return False


def _extract_user_question_from_session_history(session_history_text: str) -> str:
    """
    从 session_history_text 中提取“用户问题：...”这一行内容。
    """
    s = ensure_nonempty_str(session_history_text)
    m = re.search(r"用户问题：(.+?)(?:\n|\r|$)", s)
    if not m:
        return ""
    return (m.group(1) or "").strip()


def build_top_user_questions_module_html(
    map_results: List[Dict[str, Any]],
    min_total_pairs: int = 5,
    top_questions_each_pair: int = 10,
    max_pairs: Optional[int] = None,
) -> str:
    """
    在报告中单独放一个“用户高频真实问题清单”模块：
    - 统计每个「一级类目 + 二级类目」下“用户问题：...”出现的频率
    - 只展示出现次数 >= min_total_pairs 的组合
    - 每个组合展示出现次数 Top N 的用户原始提问（尽量过滤掉链接/图片噪声）
    """
    from collections import defaultdict

    pair_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    pair_total: Counter = Counter()
    # primary/secondary -> question -> session_id 集合（用于在表格中回溯定位）
    pair_question_session_ids: Dict[Tuple[str, str], Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))

    for it in map_results:
        llm = it.get("llm_result") or {}
        primary = ensure_nonempty_str(llm.get("primary_category")).strip()
        secondary = ensure_nonempty_str(llm.get("secondary_category")).strip()
        if not primary or not secondary:
            continue

        q = _extract_user_question_from_session_history(ensure_nonempty_str(it.get("session_history_text")))
        if not q:
            continue

        key = (primary, secondary)
        pair_counter[key][q] += 1
        pair_total[key] += 1
        sid = ensure_nonempty_str(it.get("session_id")).strip()
        if sid:
            pair_question_session_ids[key][q].add(sid)

    pairs = [(k, v) for k, v in pair_total.items() if v >= min_total_pairs]
    pairs_sorted = sorted(pairs, key=lambda x: (x[1], x[0][0], x[0][1]), reverse=True)
    if max_pairs is not None:
        pairs_sorted = pairs_sorted[: max_pairs]

    details_blocks: List[str] = []
    for (primary, secondary), total in pairs_sorted:
        c = pair_counter[(primary, secondary)]

        # 先剔除链接/图片噪声，再取 Top N
        filtered = [(q, n) for q, n in c.most_common() if not _looks_like_url_or_media(q)]
        if not filtered:
            # 若该组合全是链接/图片，就退回不过滤版本，保证模块完整可读
            filtered = list(c.most_common())

        top_qs = filtered[:top_questions_each_pair]
        rows_html: List[str] = []
        session_ids_by_q = pair_question_session_ids[(primary, secondary)]

        for idx, (q, n) in enumerate(top_qs, start=1):
            sids = sorted(list(session_ids_by_q.get(q, set())), key=lambda x: str(x))
            # 避免页面过长：默认只展示前若干条，剩余用 details 展开
            max_show = 8
            if len(sids) <= max_show:
                sids_html = f"<div style='font-family:monospace;font-size:12px;line-height:1.6;'>{html_lib.escape('，'.join(sids)) or '（无）'}</div>"
            else:
                head = html_lib.escape('，'.join(sids[:max_show]))
                tail = html_lib.escape('，'.join(sids[max_show:]))
                sids_html = (
                    f"<div style='font-family:monospace;font-size:12px;line-height:1.6;'>{head}，……（共 {len(sids)} 条）</div>"
                    f"<details style='margin-top:6px;'>"
                    f"<summary style='cursor:pointer;color:#333;font-size:12px;'>查看全部 SessionID</summary>"
                    f"<pre style='margin:6px 0 0 0;white-space:pre-wrap;word-break:break-word;font-family:monospace;font-size:12px;'>{tail}</pre>"
                    f"</details>"
                )

            rows_html.append(
                "<tr>"
                f"<td style='padding:8px;border:1px solid #eee;word-break:break-word;'>{idx}</td>"
                f"<td style='padding:8px;border:1px solid #eee;word-break:break-word;'>{n}</td>"
                f"<td style='padding:8px;border:1px solid #eee;word-break:break-word;'>{html_lib.escape(q)}</td>"
                f"<td style='padding:8px;border:1px solid #eee;word-break:break-word;'>{sids_html}</td>"
                "</tr>"
            )

        table_html = (
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<thead>"
            "<tr style='background:#f5f5f5;'>"
            "<th style='padding:8px;border:1px solid #eee;'>序号</th>"
            "<th style='padding:8px;border:1px solid #eee;'>出现次数</th>"
            "<th style='padding:8px;border:1px solid #eee;'>用户真实问题</th>"
            "<th style='padding:8px;border:1px solid #eee;'>对应 SessionID</th>"
            "</tr>"
            "</thead>"
            "<tbody>"
            + "".join(rows_html)
            + "</tbody>"
            "</table>"
        )

        details_blocks.append(
            "<details style='margin:10px 0;'>"
            f"<summary style='cursor:pointer;color:#111;font-weight:800;'>{html_lib.escape(primary)}｜{html_lib.escape(secondary)}（样本 {total} 条）</summary>"
            "<div style='margin-top:8px;'>"
            + table_html
            + "</div>"
            "</details>"
        )

    if not details_blocks:
        return (
            "<div style='padding:16px 24px;color:#666;font-size:12px;'>"
            "暂无“用户高频真实问题清单”数据（可能缺少 primary/secondary 或未解析到用户问题字段）。"
            "</div>"
        )

    return (
        "<div style='padding:16px 24px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;'>"
        "<h2 style='margin:0 0 8px 0;'>用户高频真实问题清单（按一级类目+二级类目聚合）</h2>"
        f"<div style='color:#666;font-size:12px;line-height:1.6;'>"
        f"说明：从 `tuhu_session_analysis_step6_v21_full_merged.json` 的每条样本里解析“用户问题：...”，统计每个组合出现次数；只展示出现次数 >= {min_total_pairs} 的组合，每个组合展示 Top {top_questions_each_pair}。"
        "</div>"
        "<div style='max-height:620px;overflow:auto;border:1px solid #eee;border-radius:10px;padding:12px 14px;margin-top:12px;'>"
        + "".join(details_blocks)
        + "</div>"
        "</div>"
    )


def build_report_html(
    map_results: List[Dict[str, Any]],
    task1_stats: Tuple[int, int, Dict[str, int], int],
    reduce_md: str,
    input_path: str,
    mapping_xlsx_path: str,
    report_output_html: str,
) -> None:
    total_rows, tools_rows_count, tools_counter, session_count = task1_stats

    # Task2 分布
    rewrite_counts: Dict[str, int] = {}
    answer_counts: Dict[str, int] = {}
    for item in map_results:
        llm = item.get("llm_result") or {}
        r = ensure_nonempty_str(llm.get("rewrite_issue"))
        a = ensure_nonempty_str(llm.get("answer_issue"))
        if not r:
            r = "未知"
        if not a:
            a = "未知"
        rewrite_counts[r] = rewrite_counts.get(r, 0) + 1
        answer_counts[a] = answer_counts.get(a, 0) + 1

    # Sunburst 已替换为纯 HTML 表格（在页面底部插入）

    # 图表：工具调用 top
    tools_items = sorted(tools_counter.items(), key=lambda x: x[1], reverse=True)[:15]
    tools_bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="720px", height="360px"))
    tools_bar.add_xaxis([k for k, _ in tools_items])
    tools_bar.add_yaxis(
        series_name="调用次数",
        y_axis=[v for _, v in tools_items],
    )
    tools_bar.set_global_opts(
        title_opts=opts.TitleOpts(title="Top 工具调用次数（Task 1）"),
        xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
    )

    # 图表：rewrite/answer 饼图
    rewrite_pie = Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="480px", height="360px"))
    rewrite_pie.add("", list(rewrite_counts.items()) or [("无数据", 1)], radius=["35%", "65%"])
    rewrite_pie.set_global_opts(title_opts=opts.TitleOpts(title="rewrite_issue 分布"))

    answer_pie = Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="480px", height="360px"))
    answer_pie.add("", list(answer_counts.items()) or [("无数据", 1)], radius=["35%", "65%"])
    answer_pie.set_global_opts(title_opts=opts.TitleOpts(title="answer_issue 分布"))

    # 典型坏案列表改为纯 HTML（避免 pyecharts Table 展示不全）

    # reduce markdown -> html
    reduce_html = md_lib.markdown(reduce_md, extensions=["tables"])

    # 页面拼装
    page = Page(page_title="途虎智能客服 V2.1 全局洞察报告", layout=Page.SimplePageLayout)

    # Step 1：大盘红黑榜（按 answer_issue 健康度对一级/二级类目聚合）
    pair_stats = compute_health_stats_by_category_pair(map_results)
    board_html, _red_pairs, black_pairs = build_red_black_board_html(
        pair_stats=pair_stats,
        red_topk=3,
        black_topk=5,
        min_total=5,
    )

    summary_html = f"""
    <div style="padding:16px 24px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
      <h2>途虎智能客服对话审查与全局洞察（V2.1）</h2>
      <ul>
        <li><b>原始数据总行数：</b>{total_rows}</li>
        <li><b>含 &lt;tools&gt; 标签行数：</b>{tools_rows_count}</li>
        <li><b>聚合后的会话（Session）数量：</b>{session_count}</li>
      </ul>
    </div>
    {board_html}
    """
    page.add_js_funcs(
        f"document.body.insertAdjacentHTML('afterbegin', {repr(summary_html)});"
    )

    page.add(
        tools_bar,
        rewrite_pie,
        answer_pie,
    )

    # 先渲染出基础 HTML
    page.render(report_output_html)

    # 再把 reduce_html 插入底部
    try:
        with open(report_output_html, "r", encoding="utf-8") as f:
            html = f.read()
        # 展示全部 primary->secondary 组合，并计算占比；使用滚动容器避免页面过长
        categories_table_html = build_category_pair_table_html(map_results, top_n_pairs=None)

        # 前5个一级类目下的黑榜（不需要额外 LLM）
        primary_black_modules_html = build_primary_top5_black_modules_html(
            pair_stats=pair_stats,
            top_primary=5,
            black_topk_per_primary=3,
            min_total=5,
        )

        # Step 2-3：黑榜类目专项诊断（LLM 主题聚类）+ 明细区渲染
        topic_cache_path = os.path.join(
            os.path.dirname(report_output_html),
            f"tuhu_topic_clustering_step7_v21_cache_{os.path.basename(report_output_html)}.json",
        )
        try:
            diagnosis_map = build_black_pairs_topic_diagnosis(
                map_results=map_results,
                black_pairs=black_pairs,
                topic_cache_path=topic_cache_path,
                force_topic_diagnosis=False,
                max_topic_bad_cases_chars=25000,
            )
            black_pairs_details_html = build_black_pairs_details_html(
                map_results=map_results,
                pair_stats=pair_stats,
                black_pairs=black_pairs,
                diagnosis_map=diagnosis_map,
            )
        except Exception as exc:
            print(f"警告：黑榜主题聚类诊断失败：{type(exc).__name__}: {exc}")
            black_pairs_details_html = f"""
            <div style="padding:16px 24px;color:#b00;">
              <h3 style="margin:0 0 8px 0;">黑榜专项诊断失败</h3>
              <div style="font-size:12px;line-height:1.6;">{html_lib.escape(type(exc).__name__)}：{html_lib.escape(str(exc))}</div>
            </div>
            """

        entrance_modules_html = build_entrance_modules_html(
            map_results=map_results,
            input_path=input_path,
            entrance_names=["新车品详情页入口", "保养列表页入口"],
        )
        intent_unreasonable_html = build_intent_unreasonable_table_html(
            map_results=map_results,
            mapping_xlsx_path=mapping_xlsx_path,
            min_sessions_main=5,
        )
        # 典型坏案数量可能较多；不截断展示，使用滚动容器避免页面撑爆
        typical_cases_html = build_typical_cases_table_html(map_results, limit=None)

        # 用户高频真实问题清单：从 full_merged 数据派生，方便后续换数据源重跑自动更新
        top_user_questions_module_html = build_top_user_questions_module_html(
            map_results=map_results,
            min_total_pairs=5,
            top_questions_each_pair=10,
            max_pairs=None,
        )

        insert = f"""
          {primary_black_modules_html}
          {black_pairs_details_html}
          {categories_table_html}
          {top_user_questions_module_html}
          {entrance_modules_html}
          {intent_unreasonable_html}
          {typical_cases_html}
          <div style="padding:16px 24px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
            <h2>全局系统 Prompt 迭代建议（Reduce）</h2>
            <div style="line-height:1.6;">{reduce_html}</div>
          </div>
        """
        if "</body>" in html:
            html = html.replace("</body>", insert + "</body>")
        with open(report_output_html, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception as exc:
        print(f"警告：插入 Reduce HTML 失败：{exc}")


def compute_task1_stats(input_path: str) -> Tuple[int, int, Dict[str, int], int]:
    df_raw = step1.read_input_file(input_path)
    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")
    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)
    tools_col = step1.detect_column(columns, step1.POSSIBLE_TOOLS_ANSWER_COLUMNS)

    df = step1.basic_cleaning(df_raw, session_col=session_col)
    total_rows = len(df)
    tools_rows_count, tools_counter = step1.parse_tools_usage(df, tools_col)
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    session_count = len(session_df)
    return total_rows, tools_rows_count, tools_counter, session_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Milestone 7 - V2.1 全局反思与报告生成")
    parser.add_argument("--input", "-i", required=True, help="原始对话数据文件路径")
    parser.add_argument("--mapping", "-m", default="意图描述和对应专家策略.xlsx", help="意图映射 Excel 路径")
    parser.add_argument("--strategies-dir", "-s", default=".", help="专家策略 docx 所在目录")
    parser.add_argument("--persona1", default="人设1意图识别与判断使用哪些工具.docx", help="人设1 docx")
    parser.add_argument("--persona2", default="人设2 根据工具返回的结果和常识回答问题.docx", help="人设2 docx")
    parser.add_argument("--map-output", default="tuhu_session_analysis_step6_v21_full.json", help="Map 阶段结果 JSON")
    parser.add_argument("--reduce-md", default="tuhu_global_prompt_iteration_v21.md", help="Reduce 阶段输出 Markdown 文件")
    parser.add_argument("--report-output", default="tu_hu_analysis_report_v21.html", help="最终 HTML 报告文件名")
    parser.add_argument("--map-sessions", type=int, default=-1, help="Map 阶段限制 Session 数量；-1 表示全量")
    parser.add_argument("--max-workers", type=int, default=5, help="Map 阶段最大并发线程数（建议5-10）")
    parser.add_argument("--max-retries", type=int, default=3, help="单次 LLM 调用失败重试次数")
    parser.add_argument("--force-map", action="store_true", help="强制重新跑 Map")
    parser.add_argument("--force-reduce", action="store_true", help="强制重新跑 Reduce")
    parser.add_argument("--max-bad-cases-chars", type=int, default=45000, help="Reduce 坏案拼接最大字符数")
    args = parser.parse_args()

    max_sessions: Optional[int] = None if args.map_sessions == -1 else args.map_sessions

    map_results = run_map_stage(
        input_path=args.input,
        mapping_xlsx_path=args.mapping,
        strategies_dir=args.strategies_dir,
        persona1_docx=args.persona1,
        persona2_docx=args.persona2,
        map_output_json=args.map_output,
        max_sessions=max_sessions,
        max_workers=max(1, args.max_workers),
        max_retries=max(1, args.max_retries),
        force_map=args.force_map,
    )

    # Reduce
    global_md = reduce_global_reflection(
        map_results=map_results,
        persona1_docx=args.persona1,
        persona2_docx=args.persona2,
        strategies_dir=args.strategies_dir,
        reduce_output_md=args.reduce_md,
        model_name=step2.MODEL_NAME,
        force_reduce=args.force_reduce,
        max_bad_cases_chars=args.max_bad_cases_chars,
    )

    # Task1 统计
    task1_stats = compute_task1_stats(args.input)

    # 报告生成
    build_report_html(
        map_results=map_results,
        task1_stats=task1_stats,
        reduce_md=global_md,
        input_path=args.input,
        mapping_xlsx_path=args.mapping,
        report_output_html=args.report_output,
    )

    print(f"Milestone 7 已完成，报告生成：{args.report_output}")


if __name__ == "__main__":
    main()

