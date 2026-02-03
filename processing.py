"""
processing.py - データクレンジングと分析ロジック

機能:
1. クレンジング（重複削除、欠損値補完、異常値フラグ立て）
2. 指標計算（RFM分析、初回購入商品別リピート率、LTV予測）
3. 処理ログの保持（デモ説明用）
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Any
import re

# DuckDBはオプション（インストールされている場合のみ使用）
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False


class ProcessingLog:
    """処理ログを保持するクラス"""

    def __init__(self):
        self.logs: List[Dict[str, Any]] = []

    def add(self, step: str, action: str, details: Dict[str, Any] = None):
        """ログエントリを追加"""
        self.logs.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "step": step,
            "action": action,
            "details": details or {}
        })

    def get_logs(self) -> List[Dict[str, Any]]:
        """全ログを取得"""
        return self.logs

    def get_summary(self) -> pd.DataFrame:
        """ログをDataFrameとして取得"""
        return pd.DataFrame(self.logs)


# ==============================================================================
# クレンジング関数群
# ==============================================================================

def standardize_date(date_str: str) -> str:
    """
    様々なフォーマットの日付をYYYY-MM-DD形式に統一

    対応フォーマット:
    - YYYY-MM-DD, YYYY/MM/DD
    - DD-MM-YYYY, MM/DD/YYYY
    - YYYY年MM月DD日
    """
    if pd.isna(date_str) or date_str is None:
        return None

    date_str = str(date_str).strip()

    # 各フォーマットを試行
    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%Y年%m月%d日",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None  # パースできない場合はNone


def standardize_phone(phone: str) -> str:
    """
    電話番号を統一フォーマット（XXX-XXXX-XXXX）に変換
    """
    if pd.isna(phone) or phone is None:
        return None

    # 全角を半角に変換
    phone = phone.translate(str.maketrans('０１２３４５６７８９−', '0123456789-'))

    # 数字以外を除去
    digits = re.sub(r'\D', '', phone)

    # 11桁の携帯番号の場合、フォーマット適用
    if len(digits) == 11 and digits.startswith('0'):
        return f"{digits[:3]}-{digits[3:7]}-{digits[7:]}"

    return phone  # それ以外はそのまま返す


def standardize_payment_method(method: str) -> str:
    """
    支払い方法の表記揺れを統一
    """
    if pd.isna(method) or method is None:
        return "不明"

    method = str(method).strip().lower()

    mapping = {
        "クレジットカード": "クレジットカード",
        "クレカ": "クレジットカード",
        "credit card": "クレジットカード",
        "銀行振込": "銀行振込",
        "代金引換": "代金引換",
        "コンビニ払い": "コンビニ払い",
        "paypay": "電子マネー",
    }

    for key, value in mapping.items():
        if key.lower() in method:
            return value

    return "その他"


def clean_data(df: pd.DataFrame, log: ProcessingLog) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    メインのクレンジング処理

    Returns:
        cleaned_df: クレンジング後のデータ
        flagged_df: 異常値フラグ付きデータ（元データ保持）
    """
    log.add("クレンジング開始", "処理開始", {"レコード数": len(df)})

    # 元データをコピー
    cleaned_df = df.copy()
    flagged_df = df.copy()

    # 異常値フラグカラムを追加
    flagged_df["flag_duplicate_email"] = False
    flagged_df["flag_invalid_date"] = False
    flagged_df["flag_invalid_quantity"] = False
    flagged_df["flag_outlier_amount"] = False
    flagged_df["flag_negative_value"] = False

    # -------------------------------------------------------------------------
    # Step 1: 日付フォーマットの統一
    # -------------------------------------------------------------------------
    original_dates = cleaned_df["order_date"].copy()
    cleaned_df["order_date"] = cleaned_df["order_date"].apply(standardize_date)

    # 変換できなかった日付をフラグ
    invalid_date_mask = original_dates.notna() & cleaned_df["order_date"].isna()
    flagged_df.loc[invalid_date_mask, "flag_invalid_date"] = True

    date_fixed_count = (original_dates != cleaned_df["order_date"]).sum()
    log.add("日付統一", "YYYY-MM-DD形式に統一", {
        "変換件数": int(date_fixed_count),
        "変換不可件数": int(invalid_date_mask.sum()),
        "SQL相当": "UPDATE orders SET order_date = DATE_FORMAT(order_date, '%Y-%m-%d')"
    })

    # -------------------------------------------------------------------------
    # Step 2: 電話番号フォーマットの統一
    # -------------------------------------------------------------------------
    if "phone" in cleaned_df.columns:
        cleaned_df["phone"] = cleaned_df["phone"].apply(standardize_phone)
        log.add("電話番号統一", "XXX-XXXX-XXXX形式に統一", {
            "処理件数": int(cleaned_df["phone"].notna().sum())
        })

    # -------------------------------------------------------------------------
    # Step 3: 支払い方法の統一
    # -------------------------------------------------------------------------
    if "payment_method" in cleaned_df.columns:
        original_methods = cleaned_df["payment_method"].nunique()
        cleaned_df["payment_method"] = cleaned_df["payment_method"].apply(standardize_payment_method)
        new_methods = cleaned_df["payment_method"].nunique()
        log.add("支払い方法統一", "表記揺れを統一", {
            "統一前カテゴリ数": int(original_methods),
            "統一後カテゴリ数": int(new_methods),
            "SQL相当": "UPDATE orders SET payment_method = CASE WHEN payment_method IN ('クレカ', 'credit card') THEN 'クレジットカード' ... END"
        })

    # -------------------------------------------------------------------------
    # Step 4: 数量の異常値処理
    # -------------------------------------------------------------------------
    if "quantity" in cleaned_df.columns:
        # 数値に変換
        cleaned_df["quantity"] = pd.to_numeric(cleaned_df["quantity"], errors="coerce")

        # 異常値をフラグ（負の値、0、極端に大きい値）
        invalid_qty_mask = (
            cleaned_df["quantity"].isna() |
            (cleaned_df["quantity"] <= 0) |
            (cleaned_df["quantity"] > 100)
        )
        flagged_df.loc[invalid_qty_mask, "flag_invalid_quantity"] = True

        invalid_qty_count = invalid_qty_mask.sum()

        # 異常値を中央値で補完
        median_qty = cleaned_df.loc[~invalid_qty_mask, "quantity"].median()
        cleaned_df.loc[invalid_qty_mask, "quantity"] = median_qty

        log.add("数量クレンジング", "異常値を中央値で補完", {
            "異常値件数": int(invalid_qty_count),
            "補完値（中央値）": float(median_qty),
            "SQL相当": f"UPDATE orders SET quantity = {median_qty} WHERE quantity IS NULL OR quantity <= 0 OR quantity > 100"
        })

    # -------------------------------------------------------------------------
    # Step 5: 金額の異常値処理
    # -------------------------------------------------------------------------
    if "total_amount" in cleaned_df.columns:
        cleaned_df["total_amount"] = pd.to_numeric(cleaned_df["total_amount"], errors="coerce")

        # 外れ値フラグ（100万円以上）
        outlier_mask = cleaned_df["total_amount"] >= 1000000
        flagged_df.loc[outlier_mask, "flag_outlier_amount"] = True

        # 負の値フラグ
        negative_mask = cleaned_df["total_amount"] < 0
        flagged_df.loc[negative_mask, "flag_negative_value"] = True

        # 負の値とNullを補完（price * quantityで再計算）
        invalid_amount_mask = cleaned_df["total_amount"].isna() | (cleaned_df["total_amount"] < 0)
        if "price" in cleaned_df.columns:
            cleaned_df.loc[invalid_amount_mask, "total_amount"] = (
                cleaned_df.loc[invalid_amount_mask, "price"] *
                cleaned_df.loc[invalid_amount_mask, "quantity"]
            )

        log.add("金額クレンジング", "異常値をフラグ、欠損を再計算", {
            "外れ値件数（100万以上）": int(outlier_mask.sum()),
            "負の値件数": int(negative_mask.sum()),
            "SQL相当": "UPDATE orders SET total_amount = price * quantity WHERE total_amount IS NULL OR total_amount < 0"
        })

    # -------------------------------------------------------------------------
    # Step 6: 重複メールアドレスの処理（名寄せ）
    # -------------------------------------------------------------------------
    if "email" in cleaned_df.columns:
        # 重複をフラグ
        duplicate_mask = cleaned_df.duplicated(subset=["email"], keep="first")
        flagged_df.loc[duplicate_mask, "flag_duplicate_email"] = True

        # 重複メールの顧客名を最初の登録名に統一
        email_to_name = cleaned_df.drop_duplicates(subset=["email"], keep="first").set_index("email")["customer_name"]
        cleaned_df["customer_name_unified"] = cleaned_df["email"].map(email_to_name)

        log.add("顧客名寄せ", "同一メールアドレスの顧客名を統一", {
            "重複件数": int(duplicate_mask.sum()),
            "ユニーク顧客数": int(cleaned_df["email"].nunique()),
            "SQL相当": "WITH first_names AS (SELECT email, customer_name FROM customers GROUP BY email) UPDATE orders SET customer_name = first_names.customer_name"
        })

    # -------------------------------------------------------------------------
    # Step 7: 欠損値の補完
    # -------------------------------------------------------------------------
    fill_summary = {}

    # 年齢: 中央値で補完
    if "age" in cleaned_df.columns:
        age_null_count = cleaned_df["age"].isna().sum()
        median_age = cleaned_df["age"].median()
        cleaned_df["age"] = cleaned_df["age"].fillna(median_age)
        fill_summary["age"] = {"欠損数": int(age_null_count), "補完値": float(median_age)}

    # 都道府県: "不明"で補完
    if "prefecture" in cleaned_df.columns:
        pref_null_count = cleaned_df["prefecture"].isna().sum()
        cleaned_df["prefecture"] = cleaned_df["prefecture"].fillna("不明")
        fill_summary["prefecture"] = {"欠損数": int(pref_null_count), "補完値": "不明"}

    # 性別: "不明"で補完
    if "gender" in cleaned_df.columns:
        gender_null_count = cleaned_df["gender"].isna().sum()
        cleaned_df["gender"] = cleaned_df["gender"].fillna("不明")
        fill_summary["gender"] = {"欠損数": int(gender_null_count), "補完値": "不明"}

    log.add("欠損値補完", "各カラムの欠損を補完", fill_summary)

    # -------------------------------------------------------------------------
    # 最終サマリー
    # -------------------------------------------------------------------------
    flag_cols = [col for col in flagged_df.columns if col.startswith("flag_")]
    total_flagged = flagged_df[flag_cols].any(axis=1).sum()

    log.add("クレンジング完了", "処理完了", {
        "処理後レコード数": len(cleaned_df),
        "フラグ付きレコード数": int(total_flagged),
        "残存欠損数": int(cleaned_df.isna().sum().sum())
    })

    return cleaned_df, flagged_df


# ==============================================================================
# 分析・指標計算関数群
# ==============================================================================

def calculate_rfm(df: pd.DataFrame, log: ProcessingLog,
                  analysis_date: str = None) -> pd.DataFrame:
    """
    RFM分析を実行

    R (Recency): 最終購入からの日数
    F (Frequency): 購入回数
    M (Monetary): 合計購入金額
    """
    log.add("RFM分析", "分析開始", {})

    # 分析基準日
    if analysis_date is None:
        analysis_date = datetime.now().strftime("%Y-%m-%d")

    analysis_dt = pd.to_datetime(analysis_date)

    # 有効な注文データのみ抽出
    valid_df = df[
        df["order_date"].notna() &
        df["total_amount"].notna() &
        (df["total_amount"] > 0)
    ].copy()

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"])

    # 顧客ごとに集計
    rfm_df = valid_df.groupby("customer_id").agg({
        "order_date": "max",  # 最終購入日
        "order_id": "count",  # 購入回数
        "total_amount": "sum"  # 合計金額
    }).reset_index()

    rfm_df.columns = ["customer_id", "last_purchase_date", "frequency", "monetary"]

    # Recency計算
    rfm_df["recency"] = (analysis_dt - rfm_df["last_purchase_date"]).dt.days

    # RFMスコア（1-5の5段階）
    rfm_df["R_score"] = pd.qcut(rfm_df["recency"], q=5, labels=[5, 4, 3, 2, 1], duplicates="drop")
    rfm_df["F_score"] = pd.qcut(rfm_df["frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rfm_df["M_score"] = pd.qcut(rfm_df["monetary"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5], duplicates="drop")

    # RFMスコアを数値に変換
    for col in ["R_score", "F_score", "M_score"]:
        rfm_df[col] = rfm_df[col].astype(int)

    # 総合スコア
    rfm_df["RFM_score"] = rfm_df["R_score"] + rfm_df["F_score"] + rfm_df["M_score"]

    # セグメント分類（相互排他的な定義）
    # 判定順序が重要：上から順に評価し、最初にマッチしたセグメントに分類
    def classify_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]

        # 1. 優良顧客: 最近購入・頻繁・高額（全てが高スコア）
        if r >= 4 and f >= 4 and m >= 4:
            return "優良顧客"

        # 2. 離反リスク顧客: 以前は頻繁に購入していたが最近来ていない
        #    （優先的に検出 - アクション必要）
        if r <= 2 and f >= 3:
            return "離反リスク顧客"

        # 3. 休眠顧客: 購入頻度も低く、最近も来ていない
        if r <= 2 and f <= 2:
            return "休眠顧客"

        # 4. アクティブ顧客: 最近購入しており、それなりの頻度
        #    （優良顧客の条件を満たさないがアクティブ）
        if r >= 4 and f >= 2:
            return "アクティブ顧客"

        # 5. 高額購入顧客: 金額は高いが頻度は低い（スポット購入者）
        if m >= 4 and f <= 2:
            return "高額購入顧客"

        # 6. 一般顧客: 上記のいずれにも該当しない
        return "一般顧客"

    rfm_df["segment"] = rfm_df.apply(classify_segment, axis=1)

    log.add("RFM分析", "分析完了", {
        "分析対象顧客数": len(rfm_df),
        "セグメント分布": rfm_df["segment"].value_counts().to_dict(),
        "SQL相当": """
SELECT
    customer_id,
    DATEDIFF('{analysis_date}', MAX(order_date)) as recency,
    COUNT(*) as frequency,
    SUM(total_amount) as monetary,
    NTILE(5) OVER (ORDER BY DATEDIFF('{analysis_date}', MAX(order_date)) DESC) as R_score,
    NTILE(5) OVER (ORDER BY COUNT(*)) as F_score,
    NTILE(5) OVER (ORDER BY SUM(total_amount)) as M_score
FROM orders
GROUP BY customer_id
"""
    })

    return rfm_df


def calculate_first_purchase_repeat_rate(df: pd.DataFrame, log: ProcessingLog) -> pd.DataFrame:
    """
    初回購入商品別のリピート率を計算

    「洗剤を最初に買った人は、その後もリピートしやすい」等の分析
    """
    log.add("リピート率分析", "分析開始", {})

    # 有効データのみ
    valid_df = df[
        df["order_date"].notna() &
        df["product_id"].notna()
    ].copy()

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"])

    # 顧客ごとの注文を日付順にソート
    valid_df = valid_df.sort_values(["customer_id", "order_date"])

    # 初回購入商品を特定
    first_purchases = valid_df.groupby("customer_id").first().reset_index()
    first_purchases = first_purchases[["customer_id", "product_id", "product_name", "category"]]
    first_purchases.columns = ["customer_id", "first_product_id", "first_product_name", "first_category"]

    # 顧客ごとの総購入回数
    purchase_counts = valid_df.groupby("customer_id")["order_id"].count().reset_index()
    purchase_counts.columns = ["customer_id", "total_purchases"]

    # 結合
    customer_analysis = first_purchases.merge(purchase_counts, on="customer_id")

    # リピート判定（2回以上購入でリピーター）
    customer_analysis["is_repeater"] = customer_analysis["total_purchases"] >= 2

    # 初回購入カテゴリ別のリピート率
    category_repeat = customer_analysis.groupby("first_category").agg({
        "customer_id": "count",
        "is_repeater": "sum"
    }).reset_index()
    category_repeat.columns = ["first_category", "total_customers", "repeaters"]
    category_repeat["repeat_rate"] = (category_repeat["repeaters"] / category_repeat["total_customers"] * 100).round(1)
    category_repeat = category_repeat.sort_values("repeat_rate", ascending=False)

    # 初回購入商品別のリピート率（上位）
    product_repeat = customer_analysis.groupby(["first_product_id", "first_product_name"]).agg({
        "customer_id": "count",
        "is_repeater": "sum"
    }).reset_index()
    product_repeat.columns = ["product_id", "product_name", "total_customers", "repeaters"]
    product_repeat["repeat_rate"] = (product_repeat["repeaters"] / product_repeat["total_customers"] * 100).round(1)
    product_repeat = product_repeat.sort_values("repeat_rate", ascending=False)

    log.add("リピート率分析", "分析完了", {
        "分析対象顧客数": len(customer_analysis),
        "全体リピート率": f"{customer_analysis['is_repeater'].mean() * 100:.1f}%",
        "カテゴリ別リピート率": category_repeat.set_index("first_category")["repeat_rate"].to_dict(),
        "SQL相当": """
WITH first_orders AS (
    SELECT customer_id, product_id, category,
           ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date) as rn
    FROM orders
),
customer_counts AS (
    SELECT customer_id, COUNT(*) as total_purchases
    FROM orders GROUP BY customer_id
)
SELECT
    f.category,
    COUNT(*) as total_customers,
    SUM(CASE WHEN c.total_purchases >= 2 THEN 1 ELSE 0 END) as repeaters
FROM first_orders f
JOIN customer_counts c ON f.customer_id = c.customer_id
WHERE f.rn = 1
GROUP BY f.category
"""
    })

    return {
        "customer_analysis": customer_analysis,
        "category_repeat": category_repeat,
        "product_repeat": product_repeat
    }


def calculate_ltv(df: pd.DataFrame, log: ProcessingLog,
                  months: int = 12) -> pd.DataFrame:
    """
    LTV（顧客生涯価値）予測

    簡易版: 平均購入単価 × 購入頻度 × 継続期間
    """
    log.add("LTV予測", "分析開始", {"予測期間（月）": months})

    # 有効データのみ
    valid_df = df[
        df["order_date"].notna() &
        df["total_amount"].notna() &
        (df["total_amount"] > 0)
    ].copy()

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"])

    # 顧客ごとの集計
    customer_stats = valid_df.groupby("customer_id").agg({
        "order_date": ["min", "max"],
        "total_amount": ["sum", "mean"],
        "order_id": "count"
    }).reset_index()

    customer_stats.columns = ["customer_id", "first_purchase", "last_purchase",
                              "total_spent", "avg_order_value", "order_count"]

    # アクティブ期間（月）
    customer_stats["active_months"] = (
        (customer_stats["last_purchase"] - customer_stats["first_purchase"]).dt.days / 30
    ).clip(lower=1)  # 最低1ヶ月

    # 月間購入頻度
    customer_stats["monthly_frequency"] = customer_stats["order_count"] / customer_stats["active_months"]

    # LTV予測（今後12ヶ月）
    # LTV = 平均購入単価 × 月間購入頻度 × 予測期間
    customer_stats["predicted_ltv"] = (
        customer_stats["avg_order_value"] *
        customer_stats["monthly_frequency"] *
        months
    ).round(0)

    # LTVランク（絶対値ベースの閾値）
    # 現実的なECサイトの顧客価値分布を反映
    def assign_ltv_rank(ltv):
        if ltv >= 500000:
            return "A（高）"
        elif ltv >= 300000:
            return "B"
        elif ltv >= 150000:
            return "C"
        elif ltv >= 80000:
            return "D"
        else:
            return "E（低）"

    customer_stats["ltv_rank"] = customer_stats["predicted_ltv"].apply(assign_ltv_rank)

    log.add("LTV予測", "分析完了", {
        "分析対象顧客数": len(customer_stats),
        "平均LTV": f"¥{customer_stats['predicted_ltv'].mean():,.0f}",
        "LTVランク分布": customer_stats["ltv_rank"].value_counts().to_dict(),
        "SQL相当": f"""
SELECT
    customer_id,
    AVG(total_amount) as avg_order_value,
    COUNT(*) / GREATEST(DATEDIFF(MAX(order_date), MIN(order_date)) / 30, 1) as monthly_frequency,
    AVG(total_amount) * (COUNT(*) / GREATEST(DATEDIFF(MAX(order_date), MIN(order_date)) / 30, 1)) * {months} as predicted_ltv
FROM orders
WHERE total_amount > 0
GROUP BY customer_id
"""
    })

    return customer_stats


def generate_insights(df: pd.DataFrame, rfm_df: pd.DataFrame,
                      repeat_analysis: dict, ltv_df: pd.DataFrame,
                      log: ProcessingLog) -> List[Dict[str, Any]]:
    """
    分析結果からビジネスインサイトを生成
    """
    insights = []

    # インサイト1: 高リピート率カテゴリ
    top_repeat_category = repeat_analysis["category_repeat"].iloc[0]
    insights.append({
        "title": f"「{top_repeat_category['first_category']}」購入者のリピート率が最も高い",
        "detail": f"初回購入が{top_repeat_category['first_category']}カテゴリの顧客は、{top_repeat_category['repeat_rate']}%がリピーターになっています。",
        "action": f"新規顧客には{top_repeat_category['first_category']}カテゴリの商品を初回購入として推奨することで、リピート率向上が期待できます。",
        "type": "opportunity"
    })

    # インサイト2: 離反リスク顧客
    at_risk_count = len(rfm_df[rfm_df["segment"] == "離反リスク顧客"])
    at_risk_pct = at_risk_count / len(rfm_df) * 100
    insights.append({
        "title": f"離反リスク顧客が{at_risk_pct:.1f}%存在",
        "detail": f"{at_risk_count}名の顧客が「以前は頻繁に購入していたが、最近購入がない」状態です。",
        "action": "これらの顧客に対してリテンションキャンペーン（クーポン配布、おすすめ商品メール）を実施することを推奨します。",
        "type": "warning"
    })

    # インサイト3: 高LTV顧客の特徴
    high_ltv = ltv_df[ltv_df["ltv_rank"] == "A（高）"]
    avg_high_ltv = high_ltv["predicted_ltv"].mean()
    insights.append({
        "title": f"上位20%の顧客の予測LTVは平均¥{avg_high_ltv:,.0f}",
        "detail": f"高LTV顧客{len(high_ltv)}名が、今後12ヶ月で平均¥{avg_high_ltv:,.0f}の売上貢献が見込まれます。",
        "action": "VIP顧客向けの特別プログラム（限定セール先行案内、ポイント還元率アップ）の導入を検討してください。",
        "type": "opportunity"
    })

    # インサイト4: 休眠顧客の再活性化
    dormant_count = len(rfm_df[rfm_df["segment"] == "休眠顧客"])
    insights.append({
        "title": f"休眠顧客{dormant_count}名の再活性化ポテンシャル",
        "detail": f"過去に購入履歴があるが、長期間購入のない顧客が{dormant_count}名います。",
        "action": "「お久しぶりキャンペーン」として、特別割引コードの配布を検討してください。",
        "type": "opportunity"
    })

    log.add("インサイト生成", "生成完了", {"インサイト数": len(insights)})

    return insights


# ==============================================================================
# メイン処理関数（app.pyから呼び出し用）
# ==============================================================================

def run_full_pipeline(raw_data_path: str) -> Dict[str, Any]:
    """
    全パイプラインを実行してデモ用の結果を返す

    Returns:
        dict: {
            "raw_df": 生データ,
            "cleaned_df": クレンジング後データ,
            "flagged_df": フラグ付きデータ,
            "rfm_df": RFM分析結果,
            "repeat_analysis": リピート率分析結果,
            "ltv_df": LTV予測結果,
            "insights": ビジネスインサイト,
            "log": 処理ログ
        }
    """
    log = ProcessingLog()

    # 1. データ読み込み
    log.add("データ読み込み", "CSV読み込み", {"ファイル": raw_data_path})
    raw_df = pd.read_csv(raw_data_path)
    log.add("データ読み込み", "読み込み完了", {"レコード数": len(raw_df), "カラム数": len(raw_df.columns)})

    # 2. クレンジング
    cleaned_df, flagged_df = clean_data(raw_df, log)

    # 3. RFM分析
    rfm_df = calculate_rfm(cleaned_df, log)

    # 4. リピート率分析
    repeat_analysis = calculate_first_purchase_repeat_rate(cleaned_df, log)

    # 5. LTV予測
    ltv_df = calculate_ltv(cleaned_df, log)

    # 6. インサイト生成
    insights = generate_insights(cleaned_df, rfm_df, repeat_analysis, ltv_df, log)

    return {
        "raw_df": raw_df,
        "cleaned_df": cleaned_df,
        "flagged_df": flagged_df,
        "rfm_df": rfm_df,
        "repeat_analysis": repeat_analysis,
        "ltv_df": ltv_df,
        "insights": insights,
        "log": log
    }


if __name__ == "__main__":
    # テスト実行
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    raw_data_path = os.path.join(script_dir, "raw_data.csv")

    if os.path.exists(raw_data_path):
        print("パイプライン実行中...")
        results = run_full_pipeline(raw_data_path)

        print("\n" + "=" * 60)
        print("処理ログ")
        print("=" * 60)
        for log_entry in results["log"].get_logs():
            print(f"\n[{log_entry['step']}] {log_entry['action']}")
            if log_entry["details"]:
                for key, value in log_entry["details"].items():
                    print(f"  - {key}: {value}")

        print("\n" + "=" * 60)
        print("ビジネスインサイト")
        print("=" * 60)
        for insight in results["insights"]:
            print(f"\n【{insight['title']}】")
            print(f"  {insight['detail']}")
            print(f"  → {insight['action']}")
    else:
        print(f"raw_data.csv が見つかりません: {raw_data_path}")
        print("先に data_gen.py を実行してください。")
