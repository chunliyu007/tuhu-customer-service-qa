#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step2.py

里程碑 Milestone 2：接入大模型 API 与单条/小批量 Prompt 调试 (Task 2 核心)。

功能概述：
1. 复用 step1 中的数据读取与会话聚合逻辑，从离线文件中读取数据。
2. 读取前 N 个会话（默认 5 个 Session），构造完整对话上下文：
   - 用户原始问题：dialoguesentence
   - 大模型重写问题：rewritequery
   - 客服机器人回答：answer
3. 读取本地 SOP 文件（可选，默认文件名为 tuhu_sop.txt），若不存在则使用通用汽后常识说明占位。
4. 调用兼容 OpenAI 接口规范的大模型 API（如 DeepSeek / Qwen）：
   - 在文件头部配置 API_KEY、BASE_URL、MODEL_NAME。
   - 使用严格的系统 Prompt 要求模型仅返回合法 JSON 字符串，不要带 Markdown 代码块。
5. 对前 N 个会话依次调用大模型（单线程），解析 JSON 结果并打印：
   - 原始会话内容
   - LLM 评价 JSON（格式化打印）

注意：
1. 本文件只实现 Milestone 2，不涉及并发控制、重试机制和 HTML 报告生成。
2. 并发与容错将放在 Milestone 3 中实现，避免一次性写完全部逻辑。
"""

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional

from openai import OpenAI

import step1


# ===========================
# 大模型接口配置（请在这里填写）
# ===========================

# 请根据你实际使用的服务商进行配置，例如：
# - DeepSeek:   BASE_URL = "https://api.deepseek.com/v1"
# - 阿里云 DashScope (兼容 OpenAI): BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# - 其他兼容 OpenAI 的国内服务商：参考各自文档
#
# 当前配置：使用阿里云百炼 Qwen Plus 兼容 OpenAI 接口
API_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
API_KEY: str = "sk-77f711ce9e9547a6b70368cf6f8657f5"
MODEL_NAME: str = "qwen-plus"


# ===========================
# 字段候选配置（与 step1 对齐）
# ===========================

POSSIBLE_DIALOGUE_SENTENCE_COLUMNS = step1.POSSIBLE_DIALOGUE_SENTENCE_COLUMNS
POSSIBLE_ANSWER_COLUMNS = step1.POSSIBLE_ANSWER_COLUMNS
POSSIBLE_REWRITE_QUERY_COLUMNS = step1.POSSIBLE_REWRITE_QUERY_COLUMNS


# ===========================
# 工具函数
# ===========================

def load_sop_text(sop_path: str = "tuhu_sop.txt") -> str:
    """
    读取本地 SOP 文本。

    若文件不存在，则返回一段通用的汽后常识说明，作为系统 Prompt 的占位内容。
    """
    if os.path.exists(sop_path):
        try:
            with open(sop_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    print(f"已加载本地 SOP 文件：{sop_path}")
                    return content
        except Exception as exc:  # pragma: no cover - 容错分支
            print(f"读取 SOP 文件时出错，将退回通用常识评判。错误：{exc}")

    print("未找到 tuhu_sop.txt，将使用通用汽后常识作为评判依据。")
    return (
        "你是一名熟悉汽车后市场业务的质检专家，了解轮胎、保养、机油、更换配件、"
        "车辆年检、救援拖车等常见业务流程和常识，在没有具体 SOP 文档时，"
        "请基于通用业务常识进行判断，但遇到汽配适配类问题（如轮胎/机油/配件型号是否完全匹配某车型），"
        "如果信息不足，请标记为“疑似适配疑问”，而不是直接判断为客服回答错误。"
    )


def build_client() -> OpenAI:
    """
    构造 OpenAI 兼容客户端。
    """
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        raise RuntimeError(
            "请先在 step2.py 顶部配置有效的 API_KEY 再运行本脚本。"
        )

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    return client


def sanitize_llm_output(text: str) -> str:
    """
    对大模型输出做简单清洗：
    1. 去除首尾空白。
    2. 若错误地包了一层 ```json ... ``` 或 ``` ... ```，则只保留中间部分。
    3. 尝试截取第一个 { 开始到最后一个 } 结束的部分（防御性处理）。
    """
    text = text.strip()

    # 去掉可能的 ```json / ``` 包裹
    code_block_pattern = re.compile(
        r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE
    )
    m = code_block_pattern.match(text)
    if m:
        text = m.group(1).strip()

    # 仅保留从第一个 { 到最后一个 } 之间的内容（防止前后有说明文字）
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        text = text[first : last + 1]

    return text.strip()


def call_llm_for_session(
    client: OpenAI,
    session_record: Dict[str, Any],
    conversation_rows: List[Dict[str, Optional[str]]],
    sop_text: str,
) -> Dict[str, Any]:
    """
    针对单个会话调用大模型，让其返回结构化 JSON 评价结果。
    """
    user_id = session_record.get("user_id")
    session_id = session_record.get("session_id")

    # 为 Prompt 构造一个简明的会话描述
    # conversation_rows 每个元素形如：
    # {
    #   "dialoguesentence": "...",
    #   "rewritequery": "...",
    #   "answer": "..."
    # }
    convo_for_prompt = []
    for idx, row in enumerate(conversation_rows, start=1):
        convo_for_prompt.append(
            {
                "turn_index": idx,
                "user_question": row.get("dialoguesentence") or "",
                "rewrite_query": row.get("rewritequery") or "",
                "bot_answer": row.get("answer") or "",
            }
        )

    system_prompt = (
        "你是途虎养车智能客服的对话质检专家，需要根据完整会话记录，"
        "判断大模型重写问题是否合理，以及客服机器人回答是否合理，并给出改进建议。\n\n"
        "【业务背景与要求】\n"
        f"{sop_text}\n\n"
        "特别重要：默认前提是智能客服系统已经准确掌握了用户当前绑定车辆的关键信息（品牌、车系、年款、排量等）、"
        "候选商品的适配属性，以及用户在途虎历史购买和到店服务记录。也就是说，机器人在推荐商品或确认适配时，"
        "可以基于这些后台数据进行判断，而不仅仅依赖当前这几句对话文本。\n\n"
        "【评价规则】\n"
        "1. 请综合考虑用户原始问题（dialoguesentence）、大模型重写后的查询（rewritequery）和机器人回答（answer）。\n"
        "2. 对于 rewrite：\n"
        "   - 如果重写后的查询在语义上准确表达了用户意图且无关键条件缺失，则判定为“合理”。\n"
        "   - 如果重写改变了用户意图、遗漏关键信息或引入了用户未提及的强假设，则判定为“不合理”。\n"
        "3. 对于 answer：\n"
        "   - 在默认机器人掌握车辆信息和商品适配数据的前提下，如果回答给出的结论在业务上是可能成立且不明显违背常识，"
        "     则不要轻易判定为错误；只有在结论明显违背车辆/商品基本常识，或与对话中已知信息直接矛盾时，才判定为“不合理”。\n"
        "   - 如果回答明显与问题无关、给出错误业务结论、或遗漏用户核心关切点，则判定为“不合理”。\n"
        "4. 遇到汽配适配类问题（例如：轮胎/机油/配件型号是否适配某车型），当上下文信息不足以做出 100% 结论时，"
        "   且对话中也没有体现系统已经做了 VIN 校验或适配规则校验时，可以标记为“疑似适配疑问”，并提示需要进一步确认；\n"
        "   但如果机器人只是基于后台已知车辆信息做正常推荐，且没有明显违反常识，不要一概判定为不合理。\n"
        "5. 你需要对本次会话打上一个大类标签 category（例如：轮胎类、保养类、机油类、配件适配类、活动价格类、门店服务类、其他），"
        "   以便后续统计分析。\n"
        "6. is_typical 字段用于标记该会话是否具有代表性，适合收录进典型问题案例库。\n\n"
        "【输出格式要求（极其重要）】\n"
        "你必须只返回一个合法的 JSON 字符串，不要包含任何多余说明文字，不要使用 Markdown 代码块，不要在外层包裹 ```json。\n"
        "必须严格使用 UTF-8 可表示的中文，避免输出乱码。\n"
        "JSON 的字段定义如下（字段名必须完全一致）：\n"
        "{\n"
        '  \"rewrite_issue\": \"合理\" 或 \"不合理\" 或 \"疑似适配疑问\",\n'
        '  \"answer_issue\": \"合理\" 或 \"不合理\" 或 \"疑似适配疑问\",\n'
        '  \"category\": \"轮胎类\" / \"保养类\" / \"机油类\" / \"配件适配类\" / \"活动价格类\" / \"门店服务类\" / \"其他\",\n'
        '  \"suggestion\": \"对 rewrite 和 answer 的具体修改建议，使用简洁中文\",\n'
        '  \"is_typical\": true 或 false\n'
        "}\n"
        "确保 JSON 是单个对象，不要返回数组。"
    )

    user_prompt = {
        "user_id": user_id,
        "session_id": session_id,
        "conversation": convo_for_prompt,
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                "下面是一个完整会话的结构化内容，请根据上述规则进行评估并返回 JSON：\n"
                + json.dumps(user_prompt, ensure_ascii=False, indent=2)
            ),
        },
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.1,
    )

    content = response.choices[0].message.content or ""
    content = sanitize_llm_output(content)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        # 解析失败时，返回一个兜底结构，方便调试
        print(
            f"警告：JSON 解析失败，将返回兜底结构。session_id={session_id}，错误信息：{exc}"
        )
        data = {
            "rewrite_issue": "解析失败",
            "answer_issue": "解析失败",
            "category": "其他",
            "suggestion": f"原始输出无法解析为 JSON，请人工检查。原始内容：{content}",
            "is_typical": False,
        }

    return data


def extract_conversation_rows_for_session(
    df,
    row_indices: List[int],
    dialogue_col: Optional[str],
    rewrite_col: Optional[str],
    answer_col: Optional[str],
) -> List[Dict[str, Optional[str]]]:
    """
    从原始 DataFrame 中抽取某个 session 的对话记录，按既有顺序返回。
    """
    sub = df.loc[row_indices]

    rows: List[Dict[str, Optional[str]]] = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "dialoguesentence": (
                    row[dialogue_col] if dialogue_col and dialogue_col in sub.columns else None
                ),
                "rewritequery": (
                    row[rewrite_col] if rewrite_col and rewrite_col in sub.columns else None
                ),
                "answer": (
                    row[answer_col] if answer_col and answer_col in sub.columns else None
                ),
            }
        )

    return rows


def print_session_and_result(
    idx: int,
    session_record: Dict[str, Any],
    conversation_rows: List[Dict[str, Optional[str]]],
    llm_result: Dict[str, Any],
) -> None:
    """
    在终端打印单个 Session 的原始对话与 LLM JSON 评价结果。
    """
    print("=" * 80)
    print(f"Session {idx + 1}")
    print("-" * 80)
    print(
        f"用户ID: {session_record.get('user_id')}    "
        f"SessionID: {session_record.get('session_id')}"
    )
    print("-" * 80)
    print("原始对话内容：")

    for i, row in enumerate(conversation_rows, start=1):
        print(f"\n【轮次 {i}】")
        if row.get("dialoguesentence"):
            print(f"用户问题(dialoguesentence)：{row['dialoguesentence']}")
        if row.get("rewritequery"):
            print(f"重写问题(rewritequery)：{row['rewritequery']}")
        if row.get("answer"):
            print(f"机器人回答(answer)：{row['answer']}")

    print("\nLLM 评价 JSON：")
    print(json.dumps(llm_result, ensure_ascii=False, indent=2))
    print("=" * 80 + "\n")


def main() -> None:
    """
    主函数：读取数据、聚合 Session、调用大模型处理前 N 个 Session，并打印结果。
    """
    parser = argparse.ArgumentParser(
        description="途虎智能客服对话审查 Agent - Milestone 2 (接入大模型 API 与小批量 Prompt 调试)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="输入文件路径（与 step1 相同，支持 .csv / .xls / .xlsx）。",
    )
    parser.add_argument(
        "--sessions",
        "-n",
        type=int,
        default=5,
        help="用于调试的大模型评估会话数量（默认前 5 个 Session）。",
    )
    parser.add_argument(
        "--sop",
        type=str,
        default="tuhu_sop.txt",
        help="本地 SOP 文本文件路径，默认当前目录下的 tuhu_sop.txt。",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    # 1. 读取与清洗数据（复用 step1 的逻辑）
    print(f"开始读取文件：{input_path}")
    df_raw = step1.read_input_file(input_path)
    print(f"读取完成，原始行数：{len(df_raw)}，列数：{len(df_raw.columns)}")

    columns = list(df_raw.columns)
    session_col = step1.detect_column(columns, step1.POSSIBLE_SESSION_ID_COLUMNS)
    if session_col is None:
        raise ValueError("无法识别 sessionid 字段，请检查表结构。")

    user_col = step1.detect_column(columns, step1.POSSIBLE_USER_ID_COLUMNS)

    dialogue_col = step1.detect_column(columns, POSSIBLE_DIALOGUE_SENTENCE_COLUMNS)
    rewrite_col = step1.detect_column(columns, POSSIBLE_REWRITE_QUERY_COLUMNS)
    answer_col = step1.detect_column(columns, POSSIBLE_ANSWER_COLUMNS)

    print(f"检测到 sessionid 字段名：{session_col}")
    if user_col:
        print(f"检测到 userid 字段名：{user_col}")
    else:
        print("警告：未检测到 userid 字段，将仅按 sessionid 聚合会话。")

    if dialogue_col:
        print(f"检测到 dialoguesentence 字段名：{dialogue_col}")
    else:
        print("警告：未检测到 dialoguesentence 字段，Prompt 中将缺少用户原始问题。")

    if rewrite_col:
        print(f"检测到 rewritequery 字段名：{rewrite_col}")
    else:
        print("警告：未检测到 rewritequery 字段，Prompt 中将缺少重写问题。")

    if answer_col:
        print(f"检测到 answer 字段名：{answer_col}")
    else:
        print("警告：未检测到 answer 字段，Prompt 中将缺少机器人回答。")

    df = step1.basic_cleaning(df_raw, session_col=session_col)

    # 2. 聚合 Session（复用 step1 内逻辑）
    session_df = step1.group_sessions(df, session_col=session_col, user_col=user_col)
    total_sessions = len(session_df)
    print(f"聚合完成，总会话数量：{total_sessions}")

    if total_sessions == 0:
        print("没有可用的会话数据，程序结束。")
        return

    n = max(1, min(args.sessions, total_sessions))
    print(f"即将选取前 {n} 个 Session 进行大模型评估（单线程调试）。")

    # 3. 加载 SOP 与大模型客户端
    sop_text = load_sop_text(args.sop)
    client = build_client()

    # 4. 逐个 Session 调用大模型并打印结果
    for idx in range(n):
        session_record = session_df.iloc[idx].to_dict()
        row_indices = session_record.get("row_indices") or []

        if not row_indices:
            continue

        conversation_rows = extract_conversation_rows_for_session(
            df=df,
            row_indices=row_indices,
            dialogue_col=dialogue_col,
            rewrite_col=rewrite_col,
            answer_col=answer_col,
        )

        llm_result = call_llm_for_session(
            client=client,
            session_record=session_record,
            conversation_rows=conversation_rows,
            sop_text=sop_text,
        )

        print_session_and_result(
            idx=idx,
            session_record=session_record,
            conversation_rows=conversation_rows,
            llm_result=llm_result,
        )

    print("Milestone 2 执行完毕，请检查上述 5 个会话的原始内容与 LLM JSON 评价质量。")


if __name__ == "__main__":
    main()

