"""
app.py - ECサイトデータ活用デモダッシュボード

5つのタブ構成:
1. Raw: 汚れのある生データの確認
2. Clean: クレンジング前後の比較
3. Logic: SQLやロジックの説明
4. Insights: Plotlyによる可視化
5. Action: 分析結果に基づくビジネスアクション
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# 自作モジュール
from processing import run_full_pipeline, ProcessingLog
from data_gen import generate_template_csv, get_required_columns, get_optional_columns, main as generate_demo_data

# ページ設定
st.set_page_config(
    page_title="ECデータ活用デモ",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS（ビジネス向けの清潔感のあるデザイン）
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
    }
    .insight-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .warning-card {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .action-card {
        background: #ecfdf5;
        border-left: 4px solid #10b981;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


def validate_uploaded_data(df: pd.DataFrame) -> dict:
    """
    アップロードされたデータのバリデーション

    Returns:
        dict: {
            "is_valid": bool,
            "missing_required": list,  # 不足している必須カラム
            "missing_optional": list,  # 不足しているオプションカラム
            "warnings": list,          # 警告メッセージ
            "errors": list             # エラーメッセージ
        }
    """
    required_columns = get_required_columns()
    optional_columns = get_optional_columns()

    result = {
        "is_valid": True,
        "missing_required": [],
        "missing_optional": [],
        "warnings": [],
        "errors": []
    }

    # 必須カラムのチェック
    for col in required_columns:
        if col not in df.columns:
            result["missing_required"].append(col)
            result["is_valid"] = False

    # オプションカラムのチェック
    for col in optional_columns:
        if col not in df.columns:
            result["missing_optional"].append(col)

    # エラーメッセージ生成
    if result["missing_required"]:
        result["errors"].append(
            f"必須カラムが不足しています: {', '.join(result['missing_required'])}"
        )

    # 警告メッセージ生成
    if result["missing_optional"]:
        result["warnings"].append(
            f"オプションカラムが不足しています（一部の分析が制限される可能性があります）: {', '.join(result['missing_optional'])}"
        )

    # データ量のチェック
    if len(df) == 0:
        result["is_valid"] = False
        result["errors"].append("データが空です。1件以上のデータをアップロードしてください。")

    # 基本的なデータ品質チェック
    if result["is_valid"]:
        # order_dateのチェック
        if "order_date" in df.columns:
            null_dates = df["order_date"].isna().sum()
            if null_dates > len(df) * 0.5:
                result["warnings"].append(
                    f"order_dateの欠損が多いです（{null_dates}件/{len(df)}件）。日付の分析精度に影響します。"
                )

        # total_amountのチェック
        if "total_amount" in df.columns:
            null_amounts = df["total_amount"].isna().sum()
            if null_amounts > len(df) * 0.3:
                result["warnings"].append(
                    f"total_amountの欠損が多いです（{null_amounts}件/{len(df)}件）。"
                )

    return result


@st.cache_data
def load_demo_data():
    """デモ用データの読み込みと処理（キャッシュ）"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "raw_data.csv")

    if not os.path.exists(raw_data_path):
        return None

    return run_full_pipeline(raw_data_path)


def process_uploaded_data(df: pd.DataFrame):
    """アップロードされたデータを処理"""
    import tempfile

    # 一時ファイルに保存してパイプラインを実行
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8-sig') as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    try:
        results = run_full_pipeline(temp_path)
        return results
    finally:
        # 一時ファイルを削除
        os.unlink(temp_path)


def render_header():
    """ヘッダー表示"""
    st.markdown('<p class="main-header">ECサイト データ活用プロセス デモ</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">データの「汚れ」から「アクション」までの一連のプロセスを可視化します</p>', unsafe_allow_html=True)


def render_raw_tab(results: dict):
    """Raw タブ: 生データの確認"""
    st.header("生データの確認")
    st.markdown("実際のECサイトで発生しうる「データの汚れ」を含むサンプルデータです。")

    raw_df = results["raw_df"]

    # サマリーメトリクス
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総レコード数", f"{len(raw_df):,}")
    with col2:
        st.metric("欠損値を含む行", f"{raw_df.isna().any(axis=1).sum():,}")
    with col3:
        st.metric("重複メール", f"{raw_df['email'].duplicated().sum():,}")
    with col4:
        null_pct = raw_df.isna().sum().sum() / (len(raw_df) * len(raw_df.columns)) * 100
        st.metric("欠損率", f"{null_pct:.1f}%")

    st.divider()

    # データの汚れハイライト
    st.subheader("データ品質の問題点")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### 欠損値の分布")
        null_counts = raw_df.isnull().sum()
        null_counts = null_counts[null_counts > 0].sort_values(ascending=True)
        fig = px.bar(
            x=null_counts.values,
            y=null_counts.index,
            orientation='h',
            labels={'x': '欠損数', 'y': 'カラム'},
            color=null_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=300, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### 日付フォーマットの例")
        date_samples = raw_df[raw_df["order_date"].notna()]["order_date"].head(10).tolist()
        unique_formats = list(set([str(d)[:10] if d else "" for d in date_samples]))
        st.dataframe(
            pd.DataFrame({"日付サンプル": date_samples}),
            hide_index=True,
            height=300
        )

    st.divider()

    # 生データテーブル
    st.subheader("生データプレビュー")

    # フィルタ
    col1, col2 = st.columns(2)
    with col1:
        show_nulls = st.checkbox("欠損値を含む行のみ表示")
    with col2:
        show_outliers = st.checkbox("外れ値（100万円以上）を含む行のみ表示")

    display_df = raw_df.copy()
    if show_nulls:
        display_df = display_df[display_df.isna().any(axis=1)]
    if show_outliers:
        display_df = display_df[pd.to_numeric(display_df["total_amount"], errors="coerce") >= 1000000]

    st.dataframe(
        display_df.head(100),
        use_container_width=True,
        height=400
    )
    st.caption(f"表示: {len(display_df):,} 件中 上位100件")


def render_clean_tab(results: dict):
    """Clean タブ: クレンジング前後の比較"""
    st.header("データクレンジング")
    st.markdown("汚れのあるデータを分析可能な状態に変換するプロセスを示します。")

    raw_df = results["raw_df"]
    cleaned_df = results["cleaned_df"]
    flagged_df = results["flagged_df"]
    log = results["log"]

    # Before/After メトリクス
    st.subheader("クレンジング効果")

    col1, col2, col3 = st.columns(3)
    with col1:
        before_null = raw_df.isna().sum().sum()
        after_null = cleaned_df.isna().sum().sum()
        st.metric(
            "欠損値数",
            f"{after_null:,}",
            delta=f"-{before_null - after_null:,}",
            delta_color="inverse"
        )
    with col2:
        st.metric(
            "日付フォーマット統一",
            "完了",
            delta="383件を変換"
        )
    with col3:
        st.metric(
            "支払い方法カテゴリ",
            "6種類",
            delta="-1（統一）",
            delta_color="inverse"
        )

    st.divider()

    # 処理ログ
    st.subheader("処理ステップ詳細")

    for log_entry in log.get_logs():
        if log_entry["step"] in ["クレンジング開始", "データ読み込み"]:
            continue

        with st.expander(f"**{log_entry['step']}**: {log_entry['action']}", expanded=False):
            if log_entry["details"]:
                for key, value in log_entry["details"].items():
                    if key == "SQL相当":
                        st.code(value, language="sql")
                    elif isinstance(value, dict):
                        st.json(value)
                    else:
                        st.write(f"- **{key}**: {value}")

    st.divider()

    # Before/After 比較テーブル
    st.subheader("データ比較")

    comparison_col = st.selectbox(
        "比較するカラム",
        ["order_date", "phone", "payment_method", "quantity", "total_amount"]
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Before（生データ）")
        st.dataframe(
            raw_df[[comparison_col]].head(20),
            hide_index=True,
            height=400
        )
    with col2:
        st.markdown("##### After（クレンジング後）")
        st.dataframe(
            cleaned_df[[comparison_col]].head(20),
            hide_index=True,
            height=400
        )

    # 異常値フラグのサマリー
    st.divider()
    st.subheader("異常値フラグ付きレコード")

    flag_cols = [col for col in flagged_df.columns if col.startswith("flag_")]
    flag_summary = flagged_df[flag_cols].sum()

    fig = px.bar(
        x=flag_summary.index.str.replace("flag_", ""),
        y=flag_summary.values,
        labels={"x": "フラグ種別", "y": "件数"},
        color=flag_summary.values,
        color_continuous_scale="Blues"
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


def render_logic_tab(results: dict):
    """Logic タブ: SQLやロジックの説明"""
    st.header("分析ロジック解説")
    st.markdown("各分析手法のビジネス的な意味と技術的な実装を解説します。")

    # RFM分析の説明
    st.subheader("1. RFM分析")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **RFM分析とは？**

        顧客を3つの指標でスコアリングし、セグメント分類する手法です。

        | 指標 | 意味 | ビジネス解釈 |
        |------|------|--------------|
        | **R** (Recency) | 最終購入からの日数 | 最近買った顧客ほど価値が高い |
        | **F** (Frequency) | 購入回数 | 頻繁に買う顧客ほどロイヤル |
        | **M** (Monetary) | 合計購入金額 | 高額購入者は重要顧客 |
        """)

    with col2:
        st.markdown("**セグメント分類ロジック（相互排他）**")
        st.code("""
# 判定順序が重要（上から順に評価）
if R >= 4 and F >= 4 and M >= 4:
    return "優良顧客"      # 全て高スコア
elif R <= 2 and F >= 3:
    return "離反リスク顧客"  # 要アクション！
elif R <= 2 and F <= 2:
    return "休眠顧客"
elif R >= 4 and F >= 2:
    return "アクティブ顧客"  # 最近購入あり
elif M >= 4 and F <= 2:
    return "高額購入顧客"   # スポット購入者
else:
    return "一般顧客"
        """, language="python")

    with st.expander("SQL実装例を見る"):
        st.code("""
SELECT
    customer_id,
    DATEDIFF(CURRENT_DATE, MAX(order_date)) as recency,
    COUNT(*) as frequency,
    SUM(total_amount) as monetary,
    NTILE(5) OVER (ORDER BY DATEDIFF(CURRENT_DATE, MAX(order_date)) DESC) as R_score,
    NTILE(5) OVER (ORDER BY COUNT(*)) as F_score,
    NTILE(5) OVER (ORDER BY SUM(total_amount)) as M_score
FROM orders
WHERE total_amount > 0 AND order_date IS NOT NULL
GROUP BY customer_id
        """, language="sql")

    st.divider()

    # リピート率分析
    st.subheader("2. 初回購入商品別リピート率分析")

    st.markdown("""
    **目的**: どの商品を最初に購入した顧客がリピーターになりやすいかを分析

    **ビジネス活用**: 新規顧客に対して「リピートにつながりやすい商品」を優先的に推奨
    """)

    st.code("""
-- 初回購入商品を特定
WITH first_orders AS (
    SELECT
        customer_id,
        product_id,
        category,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as purchase_order
    FROM orders
),
-- 顧客ごとの購入回数をカウント
customer_purchase_counts AS (
    SELECT customer_id, COUNT(*) as total_purchases
    FROM orders
    GROUP BY customer_id
)
-- カテゴリ別リピート率を計算
SELECT
    f.category as first_purchase_category,
    COUNT(DISTINCT f.customer_id) as total_customers,
    SUM(CASE WHEN c.total_purchases >= 2 THEN 1 ELSE 0 END) as repeaters,
    ROUND(SUM(CASE WHEN c.total_purchases >= 2 THEN 1 ELSE 0 END) * 100.0
          / COUNT(DISTINCT f.customer_id), 1) as repeat_rate_pct
FROM first_orders f
JOIN customer_purchase_counts c ON f.customer_id = c.customer_id
WHERE f.purchase_order = 1
GROUP BY f.category
ORDER BY repeat_rate_pct DESC
    """, language="sql")

    st.divider()

    # LTV予測
    st.subheader("3. LTV（顧客生涯価値）予測")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("""
        **LTV計算式**

        ```
        LTV = 平均購入単価 × 月間購入頻度 × 予測期間（月）
        ```

        **例**: 平均単価5,000円、月2回購入、12ヶ月予測
        → LTV = 5,000 × 2 × 12 = **120,000円**
        """)

    with col2:
        st.markdown("""
        **LTVランク基準（絶対値ベース）**

        | ランク | LTV閾値 | 施策例 |
        |--------|---------|--------|
        | A（高） | ≥50万円 | VIPプログラム |
        | B | ≥30万円 | ポイント還元UP |
        | C | ≥15万円 | 通常対応 |
        | D | ≥8万円 | 活性化キャンペーン |
        | E（低） | <8万円 | コスト効率重視 |
        """)


def render_insights_tab(results: dict):
    """Insights タブ: Plotlyによる可視化"""
    st.header("データインサイト")
    st.markdown("分析結果から得られた重要な発見を可視化します。")

    rfm_df = results["rfm_df"]
    repeat_analysis = results["repeat_analysis"]
    ltv_df = results["ltv_df"]
    cleaned_df = results["cleaned_df"]

    # RFMセグメント分布
    st.subheader("顧客セグメント分布（RFM分析）")

    col1, col2 = st.columns([2, 1])
    with col1:
        segment_counts = rfm_df["segment"].value_counts()
        colors = {
            "優良顧客": "#10b981",
            "アクティブ顧客": "#3b82f6",
            "高額購入顧客": "#8b5cf6",
            "一般顧客": "#6b7280",
            "離反リスク顧客": "#f59e0b",
            "休眠顧客": "#ef4444"
        }
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            color=segment_counts.index,
            color_discrete_map=colors,
            hole=0.4
        )
        fig.update_traces(textposition='outside', textinfo='label+percent')
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("##### セグメント分布")
        for segment, count in segment_counts.items():
            pct = count / len(rfm_df) * 100
            st.markdown(f"**{segment}**: {count}名 ({pct:.1f}%)")

    # セグメント定義テーブル
    st.markdown("##### セグメント定義（相互排他）")
    segment_definitions = pd.DataFrame({
        "セグメント": ["優良顧客", "離反リスク顧客", "休眠顧客", "アクティブ顧客", "高額購入顧客", "一般顧客"],
        "R（最近）": ["≥4", "≤2", "≤2", "≥4", "任意", "その他"],
        "F（頻度）": ["≥4", "≥3", "≤2", "≥2", "≤2", "その他"],
        "M（金額）": ["≥4", "任意", "任意", "任意", "≥4", "その他"],
        "特徴": [
            "全指標が高い最重要顧客",
            "以前は頻繁だが最近来ない（要アクション）",
            "購入頻度・最近度ともに低い",
            "最近も購入しており継続利用中",
            "金額は高いが頻度は低いスポット顧客",
            "上記いずれにも該当しない"
        ]
    })
    st.dataframe(segment_definitions, hide_index=True, use_container_width=True)

    st.caption("※ 判定は上から順に行われ、最初にマッチしたセグメントに分類されます（相互排他）")

    st.divider()

    # カテゴリ別リピート率
    st.subheader("初回購入カテゴリ別リピート率")

    category_repeat = repeat_analysis["category_repeat"]
    fig = px.bar(
        category_repeat,
        x="first_category",
        y="repeat_rate",
        color="repeat_rate",
        color_continuous_scale="Greens",
        labels={"first_category": "初回購入カテゴリ", "repeat_rate": "リピート率 (%)"},
        text="repeat_rate"
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    fig.update_layout(coloraxis_showscale=False, height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.info("「初回購入がこのカテゴリだった顧客は、その後もリピート購入しやすい」という傾向を示しています。")

    st.divider()

    # LTV分布
    st.subheader("顧客LTV分布")

    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            ltv_df,
            x="predicted_ltv",
            nbins=30,
            labels={"predicted_ltv": "予測LTV（円）"},
            color_discrete_sequence=["#3b82f6"]
        )
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ltv_rank_counts = ltv_df["ltv_rank"].value_counts().sort_index()
        fig = px.bar(
            x=ltv_rank_counts.index,
            y=ltv_rank_counts.values,
            labels={"x": "LTVランク", "y": "顧客数"},
            color=ltv_rank_counts.values,
            color_continuous_scale="Blues"
        )
        fig.update_layout(height=350, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # RFMスコア散布図
    st.subheader("RFMスコア相関分析")

    fig = px.scatter(
        rfm_df,
        x="recency",
        y="monetary",
        size="frequency",
        color="segment",
        color_discrete_map=colors,
        labels={
            "recency": "Recency（最終購入からの日数）",
            "monetary": "Monetary（合計購入金額）",
            "frequency": "Frequency（購入回数）"
        },
        hover_data=["customer_id", "RFM_score"]
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def render_action_tab(results: dict):
    """Action タブ: ビジネスアクション提言（データに応じて動的に生成）"""
    st.header("アクションプラン")
    st.markdown("**このデータから導かれた**具体的なビジネスアクションを提案します。")

    insights = results["insights"]
    rfm_df = results["rfm_df"]
    ltv_df = results["ltv_df"]

    # セグメント別顧客数を取得
    segment_counts = rfm_df["segment"].value_counts().to_dict() if len(rfm_df) > 0 else {}

    # ========================================================================
    # インサイトカード（動的に生成されたもの）
    # ========================================================================
    st.subheader("優先アクション")

    if len(insights) == 0:
        st.info("分析データが不足しているため、アクション提案を生成できません。")
    else:
        for i, insight in enumerate(insights):
            # タイプに応じたスタイル
            if insight["type"] == "warning":
                card_class = "warning-card"
                icon = "⚠️"
                priority_badge = "🔴 緊急"
            elif insight["type"] == "info":
                card_class = "insight-card"
                icon = "ℹ️"
                priority_badge = "⚪ 情報"
            else:
                card_class = "action-card"
                icon = "💡"
                priority = insight.get("priority", 3)
                if priority <= 2:
                    priority_badge = "🟠 重要"
                else:
                    priority_badge = "🟢 推奨"

            st.markdown(f"""
            <div class="{card_class}">
                <p style="margin-bottom: 0.3rem;"><small>{priority_badge}</small></p>
                <h4 style="margin-top: 0;">{icon} {insight['title']}</h4>
                <p>{insight['detail']}</p>
                <p><strong>推奨アクション:</strong> {insight['action']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.divider()

    # ========================================================================
    # ターゲット顧客リスト（データがある場合のみ表示）
    # ========================================================================
    st.subheader("ターゲット顧客リストの出力")

    # 表示するリストを動的に決定
    available_lists = []

    # 離反リスク顧客
    at_risk = rfm_df[rfm_df["segment"] == "離反リスク顧客"] if len(rfm_df) > 0 else pd.DataFrame()
    if len(at_risk) > 0:
        available_lists.append(("離反リスク顧客", at_risk, ["customer_id", "last_purchase_date", "frequency", "monetary", "RFM_score"], "at_risk_customers.csv"))

    # 高LTV顧客
    high_ltv = ltv_df[ltv_df["ltv_rank"] == "A（高）"] if len(ltv_df) > 0 else pd.DataFrame()
    if len(high_ltv) > 0:
        available_lists.append(("高LTV顧客", high_ltv, ["customer_id", "avg_order_value", "monthly_frequency", "predicted_ltv", "ltv_rank"], "high_ltv_customers.csv"))

    # 休眠顧客
    dormant = rfm_df[rfm_df["segment"] == "休眠顧客"] if len(rfm_df) > 0 else pd.DataFrame()
    if len(dormant) > 0:
        available_lists.append(("休眠顧客", dormant, ["customer_id", "last_purchase_date", "frequency", "monetary", "RFM_score"], "dormant_customers.csv"))

    # アクティブ顧客
    active = rfm_df[rfm_df["segment"] == "アクティブ顧客"] if len(rfm_df) > 0 else pd.DataFrame()
    if len(active) > 0:
        available_lists.append(("アクティブ顧客", active, ["customer_id", "last_purchase_date", "frequency", "monetary", "RFM_score"], "active_customers.csv"))

    # 優良顧客
    premium = rfm_df[rfm_df["segment"] == "優良顧客"] if len(rfm_df) > 0 else pd.DataFrame()
    if len(premium) > 0:
        available_lists.append(("優良顧客", premium, ["customer_id", "last_purchase_date", "frequency", "monetary", "RFM_score"], "premium_customers.csv"))

    if len(available_lists) == 0:
        st.info("ターゲットリストを生成するためのデータがありません。")
    else:
        # 2列で表示（最大4つまで）
        display_lists = available_lists[:4]
        cols = st.columns(2)

        for idx, (name, df, columns, filename) in enumerate(display_lists):
            with cols[idx % 2]:
                st.markdown(f"##### {name}リスト（{len(df)}名）")

                # 存在するカラムのみ選択
                display_cols = [c for c in columns if c in df.columns]
                display_df = df[display_cols].head(50) if len(display_cols) > 0 else df.head(50)

                if "predicted_ltv" in display_df.columns:
                    display_df = display_df.sort_values("predicted_ltv", ascending=False)
                elif "monetary" in display_df.columns:
                    display_df = display_df.sort_values("monetary", ascending=False)

                st.dataframe(display_df, hide_index=True, height=250)

                csv = df[display_cols].to_csv(index=False).encode('utf-8-sig') if len(display_cols) > 0 else df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label=f"📥 CSVダウンロード",
                    data=csv,
                    file_name=filename,
                    mime="text/csv",
                    key=f"download_{filename}"
                )

    st.divider()

    # ========================================================================
    # データに基づく施策サマリー
    # ========================================================================
    st.subheader("施策優先度サマリー")

    # 動的に施策リストを生成
    actions = []
    total_customers = len(rfm_df) if len(rfm_df) > 0 else 0

    at_risk_count = segment_counts.get("離反リスク顧客", 0)
    if at_risk_count > 0:
        actions.append({
            "施策": f"離反リスク顧客へのリテンションメール（{at_risk_count}名）",
            "効果": min(5, 2 + at_risk_count // 20),
            "工数": 2,
            "優先度": "高" if at_risk_count >= 5 else "中"
        })

    high_ltv_count = len(high_ltv) if len(ltv_df) > 0 else 0
    if high_ltv_count > 0:
        actions.append({
            "施策": f"高LTV顧客向けVIPプログラム（{high_ltv_count}名）",
            "効果": 5,
            "工数": 4,
            "優先度": "高" if high_ltv_count >= 3 else "中"
        })

    dormant_count = segment_counts.get("休眠顧客", 0)
    if dormant_count > 0:
        actions.append({
            "施策": f"休眠顧客への復帰キャンペーン（{dormant_count}名）",
            "効果": min(4, 1 + dormant_count // 10),
            "工数": 2,
            "優先度": "中" if dormant_count >= 10 else "低"
        })

    active_count = segment_counts.get("アクティブ顧客", 0)
    if active_count > 0:
        actions.append({
            "施策": f"アクティブ顧客へのクロスセル（{active_count}名）",
            "効果": min(4, 2 + active_count // 30),
            "工数": 3,
            "優先度": "中"
        })

    general_count = segment_counts.get("一般顧客", 0)
    if general_count > 0 and total_customers > 0 and general_count / total_customers >= 0.3:
        actions.append({
            "施策": f"一般顧客の育成施策（{general_count}名）",
            "効果": 3,
            "工数": 3,
            "優先度": "中"
        })

    if len(actions) == 0:
        st.info("施策を提案するためのデータが不足しています。")
    else:
        actions_df = pd.DataFrame(actions)

        fig = px.scatter(
            actions_df,
            x="工数",
            y="効果",
            size=[40] * len(actions_df),
            text="施策",
            color="優先度",
            color_discrete_map={"高": "#10b981", "中": "#f59e0b", "低": "#6b7280"}
        )
        fig.update_traces(textposition='top center')
        fig.update_layout(
            height=400,
            xaxis_title="実施工数（小→大）",
            yaxis_title="期待効果（小→大）",
            xaxis=dict(range=[0, 6]),
            yaxis=dict(range=[0, 6])
        )
        st.plotly_chart(fig, use_container_width=True)

        # 施策一覧テーブル
        st.markdown("##### 施策一覧")
        st.dataframe(actions_df, hide_index=True, use_container_width=True)

    st.divider()

    # 次のステップ
    st.subheader("次のステップ")
    st.markdown("""
    1. **上記の優先アクションを実行**: 緊急度の高いものから順に対応
    2. **ターゲットリストを活用**: CSVをダウンロードしてメール配信システムに取り込み
    3. **効果測定**: 施策実行後、再度このダッシュボードで効果を確認
    4. **継続的改善**: データを更新し、PDCAサイクルを回す
    """)


def render_data_import_section():
    """サイドバーのデータインポートセクションを描画"""

    st.markdown("---")
    st.subheader("📤 自社データを試す")

    # ステップ説明
    with st.expander("使い方", expanded=False):
        st.markdown("""
        **3ステップで自社データを分析:**

        1️⃣ **テンプレートをダウンロード**
           - 必要なカラム構成を確認

        2️⃣ **データを入力して保存**
           - Excelで開いてデータ入力
           - CSV形式で保存

        3️⃣ **ここにアップロード**
           - 自動でクレンジング・分析実行
        """)

    # テンプレートダウンロード
    st.markdown("##### 1. テンプレートダウンロード")
    template_df = generate_template_csv()
    csv_template = template_df.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 テンプレートCSVをダウンロード",
        data=csv_template,
        file_name="ec_data_template.csv",
        mime="text/csv",
        use_container_width=True
    )

    # カラム説明
    with st.expander("カラム説明を見る"):
        st.markdown("""
        **必須カラム（最低限必要）:**
        | カラム名 | 説明 | 例 |
        |---------|------|-----|
        | order_id | 注文ID | O000001 |
        | order_date | 注文日 | 2024-01-15 |
        | customer_id | 顧客ID | C00001 |
        | product_id | 商品ID | P0001 |
        | category | 商品カテゴリ | 日用品 |
        | total_amount | 注文金額 | 1000 |

        **オプションカラム（あると分析が充実）:**
        - customer_name, email, phone, prefecture, age, gender
        - product_name, price, quantity
        - payment_method, status
        """)

    st.markdown("##### 2. CSVファイルをアップロード")

    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "CSVファイルを選択",
        type=["csv"],
        help="UTF-8またはShift-JISエンコーディングに対応"
    )

    return uploaded_file


def main():
    """メインアプリケーション"""
    render_header()

    # session_stateの初期化
    if "data_source" not in st.session_state:
        st.session_state.data_source = "demo"  # "demo" or "uploaded"
    if "uploaded_data" not in st.session_state:
        st.session_state.uploaded_data = None
    if "validation_result" not in st.session_state:
        st.session_state.validation_result = None

    # サイドバー
    with st.sidebar:
        st.header("データソース")

        # データインポートセクション
        uploaded_file = render_data_import_section()

        # アップロードされたファイルの処理
        if uploaded_file is not None:
            try:
                # エンコーディングを試行
                try:
                    uploaded_df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    uploaded_df = pd.read_csv(uploaded_file, encoding='shift-jis')

                # バリデーション
                validation = validate_uploaded_data(uploaded_df)
                st.session_state.validation_result = validation

                if validation["is_valid"]:
                    st.session_state.uploaded_data = uploaded_df
                    st.session_state.data_source = "uploaded"
                    st.success(f"✅ {len(uploaded_df):,}件のデータを読み込みました")

                    # 警告表示
                    for warning in validation["warnings"]:
                        st.warning(warning)
                else:
                    st.session_state.uploaded_data = None
                    # エラー表示
                    for error in validation["errors"]:
                        st.error(error)

                    # 不足カラムの詳細表示
                    if validation["missing_required"]:
                        st.markdown("**不足している必須カラム:**")
                        for col in validation["missing_required"]:
                            st.markdown(f"- `{col}`")

            except Exception as e:
                st.error(f"ファイルの読み込みに失敗しました: {str(e)}")
                st.session_state.uploaded_data = None

        st.markdown("---")

        # データソース選択
        st.markdown("##### 使用するデータ")

        # アップロードデータがある場合は選択可能
        if st.session_state.uploaded_data is not None:
            data_source = st.radio(
                "データソースを選択",
                ["アップロードデータ", "デモデータ"],
                index=0 if st.session_state.data_source == "uploaded" else 1,
                label_visibility="collapsed"
            )
            st.session_state.data_source = "uploaded" if data_source == "アップロードデータ" else "demo"
        else:
            st.info("📊 デモデータを使用中")
            st.session_state.data_source = "demo"

        st.markdown("---")

    # データ処理
    results = None

    if st.session_state.data_source == "uploaded" and st.session_state.uploaded_data is not None:
        # アップロードデータを処理
        with st.spinner("アップロードデータを処理中..."):
            try:
                results = process_uploaded_data(st.session_state.uploaded_data)
            except Exception as e:
                st.error(f"データ処理中にエラーが発生しました: {str(e)}")
                results = None
    else:
        # デモデータを読み込み
        with st.spinner("デモデータを読み込み・処理中..."):
            results = load_demo_data()

    if results is None:
        st.error("データを読み込めませんでした。")
        st.markdown("""
        **デモデータを生成するには:**
        ```bash
        python data_gen.py
        ```
        """)
        return

    # サイドバーにデータサマリーを表示
    with st.sidebar:
        st.header("データサマリー")

        # データソース表示
        if st.session_state.data_source == "uploaded":
            st.caption("📤 アップロードデータを使用中")
        else:
            st.caption("📊 デモデータを使用中")

        st.metric("総注文数", f"{len(results['raw_df']):,}")
        st.metric("ユニーク顧客数", f"{results['cleaned_df']['customer_id'].nunique():,}")
        st.metric("商品数", f"{results['cleaned_df']['product_id'].nunique():,}")

        st.divider()

        st.markdown("##### 処理ステップ")
        st.markdown("""
        1. Raw → 生データ確認
        2. Clean → クレンジング
        3. Logic → ロジック解説
        4. Insights → 可視化
        5. Action → アクション
        """)

    # タブ
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Raw",
        "🧹 Clean",
        "🔧 Logic",
        "📊 Insights",
        "🎯 Action"
    ])

    with tab1:
        render_raw_tab(results)

    with tab2:
        render_clean_tab(results)

    with tab3:
        render_logic_tab(results)

    with tab4:
        render_insights_tab(results)

    with tab5:
        render_action_tab(results)


if __name__ == "__main__":
    main()
