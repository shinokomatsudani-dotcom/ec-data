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


def convert_to_numeric(value) -> float:
    """
    様々な形式の数値表現を浮動小数点数に変換

    対応形式:
    - 全角数字: ８９０００ → 89000
    - 通貨記号: ¥200, 2500円 → 200, 2500
    - カンマ区切り: 1,000 → 1000
    - 空白・N/A: → None
    """
    if pd.isna(value) or value is None:
        return None

    # 文字列に変換
    str_val = str(value).strip()

    # 空文字列、N/A、不明などをNoneに
    if str_val == "" or str_val.upper() in ["N/A", "NA", "NULL", "不明", "-"]:
        return None

    # 全角数字を半角に変換
    fullwidth_to_halfwidth = str.maketrans(
        '０１２３４５６７８９．，−',
        '0123456789.,-'
    )
    str_val = str_val.translate(fullwidth_to_halfwidth)

    # 通貨記号、単位、不要な文字を削除
    str_val = re.sub(r'[¥￥円$]', '', str_val)
    str_val = re.sub(r'歳$', '', str_val)  # 年齢の「歳」
    str_val = re.sub(r',', '', str_val)    # カンマ区切り
    str_val = str_val.strip()

    # 数値に変換を試みる
    try:
        return float(str_val)
    except (ValueError, TypeError):
        return None


def safe_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    DataFrameの列を安全に数値型に変換

    Args:
        df: DataFrame
        column: 変換する列名

    Returns:
        数値型に変換されたSeries
    """
    if column not in df.columns:
        return pd.Series([None] * len(df))

    return df[column].apply(convert_to_numeric)


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
        "振り込み": "銀行振込",
        "代金引換": "代金引換",
        "代引き": "代金引換",
        "コンビニ払い": "コンビニ払い",
        "paypay": "電子マネー",
        "ペイペイ": "電子マネー",
    }

    for key, value in mapping.items():
        if key.lower() in method:
            return value

    return "その他"


def standardize_category(category: str) -> str:
    """
    商品カテゴリの表記揺れを統一

    対応パターン:
    - ひらがな/カタカナ/漢字の表記揺れ
    - 英語表記
    - 全角スペースや余分な空白
    """
    if pd.isna(category) or category is None:
        return "その他"

    # 前処理: 空白除去、小文字化
    cat = str(category).strip()
    cat_lower = cat.lower()
    # 全角スペースも除去
    cat_normalized = cat.replace("　", "").replace(" ", "").lower()

    # カテゴリマッピング（キーワードベース）
    # 優先度順に定義（より具体的なものを先に）
    category_mappings = [
        # 食品系
        (["食品", "しょくひん", "食料", "フード", "food"], "食品"),

        # 日用品系
        (["日用品", "にちようひん", "日用雑貨", "生活用品", "生活雑貨", "daily"], "日用品"),

        # 家電系
        (["家電", "かでん", "電化製品", "電気製品", "electronics", "電機"], "家電"),

        # 衣類系
        (["衣類", "いるい", "衣料", "ファッション", "アパレル", "服", "fashion", "clothing"], "衣類"),

        # 美容系
        (["美容", "びよう", "コスメ", "化粧", "beauty", "cosmetic"], "美容"),
    ]

    # マッチングを試行
    for keywords, standard_name in category_mappings:
        for keyword in keywords:
            if keyword in cat_normalized:
                return standard_name

    # マッチしない場合は元の値をそのまま返す（ただし空白は整形）
    return cat.strip()


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
    # Step 3.5: カテゴリ名の統一
    # -------------------------------------------------------------------------
    if "category" in cleaned_df.columns:
        original_categories = cleaned_df["category"].nunique()
        # カテゴリ統一前の値を保存（デバッグ用）
        original_cat_values = cleaned_df["category"].value_counts().to_dict()
        cleaned_df["category"] = cleaned_df["category"].apply(standardize_category)
        new_categories = cleaned_df["category"].nunique()
        log.add("カテゴリ統一", "表記揺れを統一", {
            "統一前カテゴリ数": int(original_categories),
            "統一後カテゴリ数": int(new_categories),
            "統一後の分布": cleaned_df["category"].value_counts().to_dict(),
            "SQL相当": "UPDATE orders SET category = CASE WHEN category IN ('食品', '食料品', 'しょくひん', 'フード') THEN '食品' ... END"
        })

    # -------------------------------------------------------------------------
    # Step 4: 数量の異常値処理
    # -------------------------------------------------------------------------
    if "quantity" in cleaned_df.columns:
        # 数値に変換（全角数字、通貨記号などに対応）
        cleaned_df["quantity"] = safe_numeric_column(cleaned_df, "quantity")

        # 異常値をフラグ（負の値、0、極端に大きい値）
        invalid_qty_mask = (
            cleaned_df["quantity"].isna() |
            (cleaned_df["quantity"] <= 0) |
            (cleaned_df["quantity"] > 100)
        )
        flagged_df.loc[invalid_qty_mask, "flag_invalid_quantity"] = True

        invalid_qty_count = invalid_qty_mask.sum()

        # 中央値を計算（有効な値がない場合は1をデフォルトに）
        valid_qty = cleaned_df.loc[~invalid_qty_mask, "quantity"]
        median_qty = valid_qty.median() if len(valid_qty) > 0 else 1.0

        # 異常値を中央値で補完
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
        # 数値に変換（全角数字、通貨記号などに対応）
        cleaned_df["total_amount"] = safe_numeric_column(cleaned_df, "total_amount")

        # priceも数値に変換（再計算用）
        if "price" in cleaned_df.columns:
            cleaned_df["price"] = safe_numeric_column(cleaned_df, "price")

        # 外れ値フラグ（100万円以上）
        outlier_mask = cleaned_df["total_amount"].notna() & (cleaned_df["total_amount"] >= 1000000)
        flagged_df.loc[outlier_mask, "flag_outlier_amount"] = True

        # 負の値フラグ
        negative_mask = cleaned_df["total_amount"].notna() & (cleaned_df["total_amount"] < 0)
        flagged_df.loc[negative_mask, "flag_negative_value"] = True

        # 負の値とNullを補完（price * quantityで再計算）
        invalid_amount_mask = cleaned_df["total_amount"].isna() | (cleaned_df["total_amount"] <= 0)
        if "price" in cleaned_df.columns and "quantity" in cleaned_df.columns:
            # 両方が数値の場合のみ再計算
            can_recalc_mask = (
                invalid_amount_mask &
                cleaned_df["price"].notna() &
                cleaned_df["quantity"].notna() &
                (cleaned_df["price"] > 0) &
                (cleaned_df["quantity"] > 0)
            )
            cleaned_df.loc[can_recalc_mask, "total_amount"] = (
                cleaned_df.loc[can_recalc_mask, "price"] *
                cleaned_df.loc[can_recalc_mask, "quantity"]
            )

        # それでもNullの場合は中央値で補完
        still_invalid = cleaned_df["total_amount"].isna() | (cleaned_df["total_amount"] <= 0)
        valid_amounts = cleaned_df.loc[~still_invalid, "total_amount"]
        if len(valid_amounts) > 0:
            median_amount = valid_amounts.median()
            cleaned_df.loc[still_invalid, "total_amount"] = median_amount

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

    # 年齢: 数値変換して中央値で補完
    if "age" in cleaned_df.columns:
        # 数値に変換（全角数字、「歳」などに対応）
        cleaned_df["age"] = safe_numeric_column(cleaned_df, "age")
        age_null_count = cleaned_df["age"].isna().sum()
        valid_ages = cleaned_df["age"].dropna()
        median_age = valid_ages.median() if len(valid_ages) > 0 else 35.0
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

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"], errors="coerce")

    # 有効なデータがない場合は空のDataFrameを返す
    if len(valid_df) == 0:
        log.add("RFM分析", "分析スキップ", {"理由": "有効なデータがありません"})
        return pd.DataFrame(columns=[
            "customer_id", "last_purchase_date", "frequency", "monetary",
            "recency", "R_score", "F_score", "M_score", "RFM_score", "segment"
        ])

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
    # データ数が少ない場合はqcut()が失敗するため、エラーハンドリングを追加
    def safe_qcut(series, q, labels, ascending=True):
        """安全にqcutを実行（データ数が少ない場合に対応）"""
        try:
            if len(series.unique()) < q:
                # ユニーク値が少ない場合は、利用可能な分位数で分割
                n_bins = min(len(series.unique()), q)
                if n_bins <= 1:
                    return pd.Series([labels[len(labels)//2]] * len(series), index=series.index)
                adjusted_labels = labels[:n_bins] if ascending else labels[-n_bins:]
                return pd.qcut(series.rank(method="first"), q=n_bins, labels=adjusted_labels, duplicates="drop")
            return pd.qcut(series.rank(method="first") if not ascending else series, q=q, labels=labels, duplicates="drop")
        except (ValueError, IndexError):
            # それでも失敗した場合は中央値を返す
            return pd.Series([labels[len(labels)//2]] * len(series), index=series.index)

    rfm_df["R_score"] = safe_qcut(rfm_df["recency"], q=5, labels=[5, 4, 3, 2, 1], ascending=False)
    rfm_df["F_score"] = safe_qcut(rfm_df["frequency"], q=5, labels=[1, 2, 3, 4, 5], ascending=True)
    rfm_df["M_score"] = safe_qcut(rfm_df["monetary"], q=5, labels=[1, 2, 3, 4, 5], ascending=True)

    # RFMスコアを数値に変換
    for col in ["R_score", "F_score", "M_score"]:
        rfm_df[col] = pd.to_numeric(rfm_df[col], errors="coerce").fillna(3).astype(int)

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

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"], errors="coerce")

    # 有効なデータがない場合は空の結果を返す
    if len(valid_df) == 0:
        log.add("リピート率分析", "分析スキップ", {"理由": "有効なデータがありません"})
        return {
            "customer_analysis": pd.DataFrame(columns=["customer_id", "first_product_id", "first_product_name", "first_category", "total_purchases", "is_repeater"]),
            "category_repeat": pd.DataFrame(columns=["first_category", "total_customers", "repeaters", "repeat_rate"]),
            "product_repeat": pd.DataFrame(columns=["product_id", "product_name", "total_customers", "repeaters", "repeat_rate"])
        }

    # 顧客ごとの注文を日付順にソート
    valid_df = valid_df.sort_values(["customer_id", "order_date"])

    # 初回購入商品を特定
    first_purchases = valid_df.groupby("customer_id").first().reset_index()

    # 必要なカラムを選択（存在しない場合はデフォルト値）
    cols_to_select = ["customer_id", "product_id"]
    first_purchases_data = {"customer_id": first_purchases["customer_id"], "first_product_id": first_purchases["product_id"]}

    if "product_name" in first_purchases.columns:
        first_purchases_data["first_product_name"] = first_purchases["product_name"]
    else:
        first_purchases_data["first_product_name"] = "不明"

    if "category" in first_purchases.columns:
        first_purchases_data["first_category"] = first_purchases["category"]
    else:
        first_purchases_data["first_category"] = "不明"

    first_purchases = pd.DataFrame(first_purchases_data)

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

    valid_df["order_date"] = pd.to_datetime(valid_df["order_date"], errors="coerce")

    # 有効なデータがない場合は空のDataFrameを返す
    if len(valid_df) == 0:
        log.add("LTV予測", "分析スキップ", {"理由": "有効なデータがありません"})
        return pd.DataFrame(columns=[
            "customer_id", "first_purchase", "last_purchase", "total_spent",
            "avg_order_value", "order_count", "active_months", "monthly_frequency",
            "predicted_ltv", "ltv_rank"
        ])

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
    分析結果からビジネスインサイトを動的に生成

    データの特性に応じて、関連性の高いインサイトのみを生成する
    """
    insights = []
    total_customers = len(rfm_df) if len(rfm_df) > 0 else 1

    # セグメント別の顧客数を取得
    segment_counts = rfm_df["segment"].value_counts().to_dict() if len(rfm_df) > 0 else {}

    # ========================================================================
    # 優先度1: 緊急アクションが必要なインサイト（警告）
    # ========================================================================

    # 離反リスク顧客（5%以上かつ5名以上の場合のみ表示）
    at_risk_count = segment_counts.get("離反リスク顧客", 0)
    at_risk_pct = at_risk_count / total_customers * 100
    if at_risk_count >= 5 and at_risk_pct >= 5:
        insights.append({
            "title": f"離反リスク顧客が{at_risk_count}名（{at_risk_pct:.1f}%）",
            "detail": f"以前は頻繁に購入していたが最近購入がない顧客です。放置すると完全離脱の可能性があります。",
            "action": "【優先度：高】リテンションメール（クーポン付き）を送信し、購入を促してください。",
            "type": "warning",
            "priority": 1,
            "impact_score": at_risk_count * 10  # インパクトスコア
        })

    # ========================================================================
    # 優先度2: 成長機会のインサイト
    # ========================================================================

    # 高リピート率カテゴリ（データがある場合のみ）
    category_repeat = repeat_analysis.get("category_repeat", pd.DataFrame())
    if len(category_repeat) > 0:
        # 顧客数が10名以上のカテゴリのみ対象
        significant_categories = category_repeat[category_repeat["total_customers"] >= 10]
        if len(significant_categories) > 0:
            top_cat = significant_categories.iloc[0]
            # 全体平均リピート率を計算
            overall_repeat = repeat_analysis["customer_analysis"]["is_repeater"].mean() * 100 if len(repeat_analysis.get("customer_analysis", [])) > 0 else 0
            if top_cat["repeat_rate"] > overall_repeat:
                insights.append({
                    "title": f"「{top_cat['first_category']}」購入者のリピート率が{top_cat['repeat_rate']:.1f}%",
                    "detail": f"このカテゴリの初回購入者は全体平均（{overall_repeat:.1f}%）より高いリピート率を示しています。",
                    "action": f"新規顧客獲得キャンペーンで「{top_cat['first_category']}」を推奨商品として訴求してください。",
                    "type": "opportunity",
                    "priority": 2,
                    "impact_score": top_cat["total_customers"] * (top_cat["repeat_rate"] - overall_repeat) / 10
                })

    # 高LTV顧客（存在する場合のみ）
    if len(ltv_df) > 0:
        high_ltv = ltv_df[ltv_df["ltv_rank"] == "A（高）"]
        if len(high_ltv) >= 3:
            avg_high_ltv = high_ltv["predicted_ltv"].mean()
            insights.append({
                "title": f"高LTV顧客{len(high_ltv)}名（平均¥{avg_high_ltv:,.0f}）",
                "detail": f"上位顧客は今後12ヶ月で大きな売上貢献が見込まれます。",
                "action": "VIPプログラム（限定セール先行案内、ポイント還元率UP）で囲い込みを強化してください。",
                "type": "opportunity",
                "priority": 2,
                "impact_score": len(high_ltv) * avg_high_ltv / 100000
            })

    # アクティブ顧客が多い場合（ポジティブなインサイト）
    active_count = segment_counts.get("アクティブ顧客", 0)
    active_pct = active_count / total_customers * 100
    if active_count >= 10 and active_pct >= 20:
        insights.append({
            "title": f"アクティブ顧客が{active_count}名（{active_pct:.1f}%）",
            "detail": f"最近も購入しており、継続利用している健全な顧客層です。",
            "action": "クロスセル施策（関連商品レコメンド）で客単価向上を狙ってください。",
            "type": "opportunity",
            "priority": 3,
            "impact_score": active_count * 5
        })

    # ========================================================================
    # 優先度3: 改善余地のあるインサイト
    # ========================================================================

    # 休眠顧客（10名以上かつ10%以上の場合のみ）
    dormant_count = segment_counts.get("休眠顧客", 0)
    dormant_pct = dormant_count / total_customers * 100
    if dormant_count >= 10 and dormant_pct >= 10:
        insights.append({
            "title": f"休眠顧客{dormant_count}名（{dormant_pct:.1f}%）の再活性化余地",
            "detail": f"過去に購入履歴があるが、長期間購入のない顧客です。",
            "action": "「お久しぶりキャンペーン」（特別割引コード）で復帰を促してください。",
            "type": "opportunity",
            "priority": 3,
            "impact_score": dormant_count * 3
        })

    # 高額購入顧客（スポット購入者）が多い場合
    high_value_count = segment_counts.get("高額購入顧客", 0)
    high_value_pct = high_value_count / total_customers * 100
    if high_value_count >= 5 and high_value_pct >= 5:
        insights.append({
            "title": f"高額スポット購入者{high_value_count}名（{high_value_pct:.1f}%）",
            "detail": f"購入頻度は低いが、購入時の金額が高い顧客層です。",
            "action": "購入後フォローメール（使い方ガイド、関連商品）で継続購入を促進してください。",
            "type": "opportunity",
            "priority": 3,
            "impact_score": high_value_count * 8
        })

    # 一般顧客が大半を占める場合
    general_count = segment_counts.get("一般顧客", 0)
    general_pct = general_count / total_customers * 100
    if general_pct >= 40:
        insights.append({
            "title": f"一般顧客が{general_pct:.1f}%を占めている",
            "detail": f"特定セグメントに分類されない顧客が多く、育成余地があります。",
            "action": "購入頻度向上キャンペーン（リピート購入で割引）を検討してください。",
            "type": "opportunity",
            "priority": 4,
            "impact_score": general_count * 2
        })

    # ========================================================================
    # 優先度4: 低リピート率カテゴリの改善
    # ========================================================================
    if len(category_repeat) > 0:
        low_repeat_cats = category_repeat[
            (category_repeat["total_customers"] >= 10) &
            (category_repeat["repeat_rate"] < 50)
        ]
        if len(low_repeat_cats) > 0:
            worst_cat = low_repeat_cats.iloc[-1]
            insights.append({
                "title": f"「{worst_cat['first_category']}」のリピート率が{worst_cat['repeat_rate']:.1f}%と低い",
                "detail": f"このカテゴリの初回購入者はリピーターになりにくい傾向があります。",
                "action": f"初回購入後のフォローアップ（関連商品提案、使用レビュー依頼）を強化してください。",
                "type": "warning",
                "priority": 4,
                "impact_score": worst_cat["total_customers"] * (50 - worst_cat["repeat_rate"]) / 10
            })

    # ========================================================================
    # インサイトがない場合のフォールバック
    # ========================================================================
    if len(insights) == 0:
        insights.append({
            "title": "データ量が少なく、明確なインサイトを抽出できません",
            "detail": "より多くの注文データを蓄積することで、詳細な分析が可能になります。",
            "action": "継続的なデータ収集を行い、再度分析を実施してください。",
            "type": "info",
            "priority": 5,
            "impact_score": 0
        })

    # 優先度（昇順：1が最優先）→インパクトスコア（降順）でソート
    insights = sorted(insights, key=lambda x: (x.get("priority", 5), -x.get("impact_score", 0)))

    # 上位5件に制限
    insights = insights[:5]

    log.add("インサイト生成", "生成完了", {
        "インサイト数": len(insights),
        "セグメント分布": segment_counts
    })

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
