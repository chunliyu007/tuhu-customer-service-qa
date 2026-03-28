#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
step4.py

里程碑 Milestone 4：HTML 数据可视化分析报告生成。

功能概述：
1. 复用 step1 的 Task 1 统计逻辑，对输入文件做定量分析：
   - 总数据行数
   - 含 <tools> 标签的行数
   - 各工具名调用次数分布
   - 聚合后的 Session 数量
2. 读取 step3 跑批生成的 JSON 结果文件（每个 Session 一条记录），对 Task 2 结果做统计：
   - rewrite_issue 分布
   - answer_issue 分布
   - category 分布
3. 从中筛选出“典型问题 SessionID”：
   - is_typical == true
   - 且（rewrite_issue 或 answer_issue 至少有一个不是“合理”）
4. 使用 pyecharts 在本地生成带图表的 HTML 报告：
   - 工具调用次数柱状图 / 饼图
   - Task 2 评价结果的饼图（category、rewrite_issue、answer_issue）
   - 典型问题 Session 列表（表格形式）
5. 输出文件名：tu_hu_analysis_report.html。

注意：
1. 本脚本只做可视化与 HTML 生成，不再调用大模型。
2. 不修改之前步骤的任何逻辑，只做读取与展示。
"""

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

from pyecharts import options as opts
from pyecharts.charts import Bar, Grid, Page, Pie
from pyecharts.components import Table
from pyecharts.globals import ThemeType

import step1


def compute_task1_stats(input_path: str) -> Tuple[int, int, Dict[str, int], int]:
    """
    复用 step1 的逻辑，计算 Task 1 相关统计。
    """
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


def load_task2_results(json_path: str) -> List[Dict[str, Any]]:
    """
    读取 step3 生成的 JSON 结果文件。
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"未找到批处理结果文件：{json_path} ，请先运行 step3.py 生成该文件。"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("JSON 结果格式异常：顶层应为数组。")

    return data


def summarize_task2(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对 Task 2 的结果做聚合统计。
    返回字典包含：
        - rewrite_issue_counter
        - answer_issue_counter
        - category_counter
        - typical_issues: 典型问题 Session 列表
    """
    rewrite_issue_counter: Dict[str, int] = {}
    answer_issue_counter: Dict[str, int] = {}
    category_counter: Dict[str, int] = {}
    typical_issues: List[Dict[str, Any]] = []

    for item in results:
        if "error" in item:
            # 失败的 Session 略过统计
            continue

        llm_result = item.get("llm_result") or {}
        rewrite_issue = str(llm_result.get("rewrite_issue", "未知"))
        answer_issue = str(llm_result.get("answer_issue", "未知"))
        category = str(llm_result.get("category", "其他"))
        is_typical = bool(llm_result.get("is_typical", False))

        rewrite_issue_counter[rewrite_issue] = (
            rewrite_issue_counter.get(rewrite_issue, 0) + 1
        )
        answer_issue_counter[answer_issue] = (
            answer_issue_counter.get(answer_issue, 0) + 1
        )
        category_counter[category] = category_counter.get(category, 0) + 1

        # 典型问题：标记为典型，且存在至少一项不为“合理”
        if is_typical and (rewrite_issue != "合理" or answer_issue != "合理"):
            typical_issues.append(
                {
                    "user_id": item.get("user_id"),
                    "session_id": item.get("session_id"),
                    "rewrite_issue": rewrite_issue,
                    "answer_issue": answer_issue,
                    "category": category,
                    "suggestion": str(llm_result.get("suggestion", "")),
                }
            )

    return {
        "rewrite_issue_counter": rewrite_issue_counter,
        "answer_issue_counter": answer_issue_counter,
        "category_counter": category_counter,
        "typical_issues": typical_issues,
    }


def make_pie(title: str, data: Dict[str, int]) -> Pie:
    items = [(k, v) for k, v in data.items() if v > 0]
    if not items:
        items = [("无数据", 1)]

    c = (
        Pie(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="480px", height="360px"))
        .add("", items, radius=["35%", "65%"])
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            legend_opts=opts.LegendOpts(orient="vertical", pos_right="5%", pos_top="20%"),
        )
        .set_series_opts(
            label_opts=opts.LabelOpts(formatter="{b}: {c} ({d}%)")
        )
    )
    return c


def make_bar(title: str, data: Dict[str, int]) -> Bar:
    items = sorted(data.items(), key=lambda x: x[1], reverse=True)
    if not items:
        items = [("无数据", 0)]

    names = [k for k, _ in items]
    values = [v for _, v in items]

    c = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.LIGHT, width="720px", height="360px"))
        .add_xaxis(names)
        .add_yaxis("次数", values)
        .set_global_opts(
            title_opts=opts.TitleOpts(title=title),
            xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=30)),
            datazoom_opts=[opts.DataZoomOpts(type_="slider")],
        )
    )
    return c


def make_typical_table(typical_issues: List[Dict[str, Any]]) -> Table:
    table = Table()
    headers = ["UserID", "SessionID", "类别", "重写问题评价", "回答评价", "修改建议"]
    rows: List[List[str]] = []
    for item in typical_issues:
        rows.append(
            [
                str(item.get("user_id") or ""),
                str(item.get("session_id") or ""),
                str(item.get("category") or ""),
                str(item.get("rewrite_issue") or ""),
                str(item.get("answer_issue") or ""),
                str(item.get("suggestion") or ""),
            ]
        )

    if not rows:
        rows.append(["-", "-", "-", "-", "-", "暂无被标记为典型问题的会话。"])

    table.add(headers, rows)
    table.set_global_opts(
        title_opts=opts.ComponentTitleOpts(title="典型问题 Session 列表（需要重点优化）")
    )
    return table


def build_report_page(
    task1_stats: Tuple[int, int, Dict[str, int], int],
    task2_summary: Dict[str, Any],
) -> Page:
    total_rows, tools_rows_count, tools_counter, session_count = task1_stats
    rewrite_issue_counter = task2_summary["rewrite_issue_counter"]
    answer_issue_counter = task2_summary["answer_issue_counter"]
    category_counter = task2_summary["category_counter"]
    typical_issues = task2_summary["typical_issues"]

    page = Page(page_title="途虎智能客服对话审查与洞察报告")

    # Task 1 图表
    tools_bar = make_bar("工具调用次数分布（Task 1）", tools_counter)
    tools_pie = make_pie("含 <tools> 行 / 总行数 概览", {
        "含 <tools> 标签行数": tools_rows_count,
        "其他行数": max(0, total_rows - tools_rows_count),
    })

    # Task 2 图表
    rewrite_pie = make_pie("重写合理性分布（rewrite_issue）", rewrite_issue_counter)
    answer_pie = make_pie("回答合理性分布（answer_issue）", answer_issue_counter)
    category_pie = make_pie("问题类别分布（category）", category_counter)

    typical_table = make_typical_table(typical_issues)

    # 使用 Grid 对部分图表做简单布局
    grid_top = Grid()
    grid_top.add(tools_bar, grid_opts=opts.GridOpts(pos_bottom="55%"))
    grid_top.add(tools_pie, grid_opts=opts.GridOpts(pos_top="55%"))

    grid_mid = Grid()
    grid_mid.add(rewrite_pie, grid_opts=opts.GridOpts(pos_left="0%", pos_right="66%"))
    grid_mid.add(answer_pie, grid_opts=opts.GridOpts(pos_left="33%", pos_right="33%"))
    grid_mid.add(category_pie, grid_opts=opts.GridOpts(pos_left="66%", pos_right="0%"))

    page.add(grid_top, grid_mid, typical_table)

    # 在页面顶部添加简要文字说明（通过自定义 HTML）
    summary_html = f"""
    <div style="padding:16px 24px;font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;">
      <h2>途虎智能客服对话审查与洞察报告</h2>
      <p>本报告基于离线对话数据与大模型审查结果自动生成，用于帮助产品经理和运营同学快速发现智能客服在工具调用、问答重写与业务回复上的优化空间。</p>
      <ul>
        <li><b>原始数据总行数：</b>{total_rows}</li>
        <li><b>含 &lt;tools&gt; 标签行数：</b>{tools_rows_count}</li>
        <li><b>聚合后的会话（Session）数量：</b>{session_count}</li>
        <li><b>已审查的会话数量：</b>{sum(rewrite_issue_counter.values())}</li>
        <li><b>被标记为典型问题的会话数量：</b>{len(typical_issues)}</li>
      </ul>
      <p style="color:#666;font-size:12px;">说明：图表为静态本地渲染，无需联网即可查看。若需进一步钻取，可结合 JSON 结果与原始日志进行人工复盘。</p>
    </div>
    """
    page.add_js_funcs(
        f"document.body.insertAdjacentHTML('afterbegin', {repr(summary_html)});"
    )

    return page


def main() -> None:
    parser = argparse.ArgumentParser(
        description="途虎智能客服对话审查 Agent - Milestone 4 (HTML 报告生成)"
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="原始对话数据文件路径（与 step1/step3 相同，支持 .csv / .xls / .xlsx）。",
    )
    parser.add_argument(
        "--json",
        "-j",
        default="tuhu_session_analysis_step3.json",
        help="step3 生成的批量审查结果 JSON 文件路径，默认 tuhu_session_analysis_step3.json。",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="tu_hu_analysis_report.html",
        help="输出 HTML 报告文件名，默认 tu_hu_analysis_report.html。",
    )

    args = parser.parse_args()

    input_path = args.input
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    json_path = args.json
    output_path = args.output

    print("开始计算 Task 1 统计指标...")
    task1_stats = compute_task1_stats(input_path)

    print("开始加载 Task 2 JSON 结果...")
    task2_results = load_task2_results(json_path)
    task2_summary = summarize_task2(task2_results)

    print("开始生成 HTML 报告...")
    page = build_report_page(task1_stats, task2_summary)
    page.render(output_path)

    print(f"报告已生成：{output_path} ，可以直接在浏览器中打开查看。")


if __name__ == "__main__":
    main()

