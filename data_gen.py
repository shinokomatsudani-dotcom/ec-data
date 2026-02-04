"""
data_gen.py - ECサイトの「現実的な汚れ」を含むモックデータ生成スクリプト

生成されるデータの汚れ:
- 顧客名の重複（同じメアドで別名登録）
- 注文日のフォーマット不備、欠損値
- 極端な外れ値（高額注文）
- 商品名の表記揺れ
- 電話番号のフォーマット不統一
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# 再現性のためのシード設定
np.random.seed(42)
random.seed(42)


def generate_customers(n_customers: int = 200) -> pd.DataFrame:
    """
    顧客マスタを生成（意図的な汚れを含む）

    汚れの種類:
    - 同じメールアドレスで異なる名前（名前の表記揺れ）
    - 電話番号のフォーマット不統一
    - 一部の顧客情報が欠損
    """

    # 基本的な名前リスト
    last_names = ["田中", "鈴木", "佐藤", "山田", "渡辺", "伊藤", "中村", "小林", "加藤", "吉田",
                  "山本", "松本", "井上", "木村", "林", "斎藤", "清水", "山崎", "森", "池田"]
    first_names = ["太郎", "花子", "一郎", "美咲", "健太", "さくら", "翔太", "愛", "大輔", "結衣",
                   "拓海", "陽菜", "蓮", "美月", "悠斗", "葵", "翼", "杏", "颯太", "凛"]

    domains = ["gmail.com", "yahoo.co.jp", "docomo.ne.jp", "softbank.ne.jp", "icloud.com"]
    prefectures = ["東京都", "大阪府", "神奈川県", "愛知県", "埼玉県", "千葉県", "兵庫県", "北海道", "福岡県", "静岡県"]

    customers = []
    used_emails = {}

    for i in range(n_customers):
        customer_id = f"C{str(i+1).zfill(5)}"
        last_name = random.choice(last_names)
        first_name = random.choice(first_names)
        full_name = f"{last_name} {first_name}"

        # メールアドレス生成
        email_base = f"{last_name.lower()}{first_name.lower()}{random.randint(1, 99)}"
        domain = random.choice(domains)
        email = f"{email_base}@{domain}"

        # 汚れ1: 約10%の確率で既存メールアドレスを再利用（名前違い）
        if i > 20 and random.random() < 0.10:
            existing_email = random.choice(list(used_emails.keys()))
            email = existing_email
            # 名前を微妙に変える（表記揺れ）
            original_name = used_emails[existing_email]
            name_variations = [
                original_name.replace(" ", "　"),  # 全角スペース
                original_name.replace(" ", ""),    # スペースなし
                f"{original_name}（旧姓）",
                original_name + " ",               # 末尾スペース
            ]
            full_name = random.choice(name_variations)
        else:
            used_emails[email] = full_name

        # 電話番号生成（フォーマット不統一）
        phone_formats = [
            f"090-{random.randint(1000,9999)}-{random.randint(1000,9999)}",
            f"080{random.randint(10000000,99999999)}",
            f"070 {random.randint(1000,9999)} {random.randint(1000,9999)}",
            f"０９０−{random.randint(1000,9999)}−{random.randint(1000,9999)}",  # 全角
            None,  # 欠損
        ]
        phone = random.choices(phone_formats, weights=[0.4, 0.25, 0.15, 0.1, 0.1])[0]

        # 登録日
        register_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 730))

        # 汚れ2: 一部の住所が欠損
        prefecture = random.choice(prefectures) if random.random() > 0.05 else None

        customers.append({
            "customer_id": customer_id,
            "customer_name": full_name,
            "email": email,
            "phone": phone,
            "prefecture": prefecture,
            "register_date": register_date.strftime("%Y-%m-%d"),
            "age": random.randint(18, 75) if random.random() > 0.08 else None,
            "gender": random.choice(["M", "F", "その他", None])
        })

    return pd.DataFrame(customers)


def generate_products(n_products: int = 50) -> pd.DataFrame:
    """
    商品マスタを生成（表記揺れを含む）
    """

    categories = {
        "日用品": ["洗剤", "シャンプー", "ティッシュ", "トイレットペーパー", "歯磨き粉", "石鹸", "柔軟剤"],
        "食品": ["お米", "パスタ", "調味料セット", "缶詰", "インスタント麺", "お菓子詰め合わせ", "コーヒー"],
        "家電": ["電気ケトル", "加湿器", "扇風機", "電子レンジ", "トースター", "掃除機", "ドライヤー"],
        "ファッション": ["Tシャツ", "ジーンズ", "スニーカー", "バッグ", "財布", "帽子", "サングラス"],
        "美容": ["化粧水", "乳液", "日焼け止め", "リップクリーム", "ハンドクリーム", "美容液", "フェイスマスク"],
    }

    products = []
    product_id = 1

    for category, items in categories.items():
        for item in items:
            # 基本商品
            base_price = random.randint(300, 15000)

            # 汚れ: 商品名の表記揺れ（同じ商品が複数登録されているケース）
            name_variations = [item]
            if random.random() < 0.3:
                name_variations.extend([
                    f"{item}（お徳用）",
                    f"【セール】{item}",
                    item.replace("ー", "-"),  # 長音の表記揺れ
                ])

            for name in name_variations:
                products.append({
                    "product_id": f"P{str(product_id).zfill(4)}",
                    "product_name": name,
                    "category": category,
                    "price": base_price + random.randint(-100, 500),
                    "cost": int(base_price * random.uniform(0.3, 0.6)),
                })
                product_id += 1

                if product_id > n_products:
                    break
        if product_id > n_products:
            break

    return pd.DataFrame(products)


def generate_orders(customers_df: pd.DataFrame, products_df: pd.DataFrame,
                    n_orders: int = 2000) -> pd.DataFrame:
    """
    注文データを生成（様々な汚れを含む）

    汚れの種類:
    - 日付フォーマットの不統一
    - 欠損値（数量、金額など）
    - 極端な外れ値（高額注文）
    - 不正な値（マイナス数量など）

    現実的な特性:
    - 購入回数はべき乗分布（少数のヘビーユーザー、多数のライトユーザー）
    - 約30%の顧客は1回のみ購入（リピートなし）
    """

    customer_ids = customers_df["customer_id"].tolist()
    product_ids = products_df["product_id"].tolist()
    product_prices = dict(zip(products_df["product_id"], products_df["price"]))

    # 顧客ごとの購入回数を事前に決定（べき乗分布 + 1回のみ顧客）
    n_customers = len(customer_ids)

    # 顧客タイプを割り当て（セグメント分布を多様化するため）
    # - dormant: 休眠顧客（古い購入日 + 低頻度）
    # - high_value: 高額購入顧客（高額 + 低頻度）
    # - loyal: 優良顧客候補（高頻度 + 最近も購入）
    # - churn_risk: 離反リスク（以前は頻繁だが最近来ない）
    # - normal: 通常パターン
    #
    # 注意: 休眠顧客と高額購入顧客はどちらか一方のみ生成
    # 時間ベースのランダム選択（固定シードの影響を受けない）
    import time
    special_segment = ["dormant", "high_value"][int(time.time() * 1000) % 2]

    customer_types = []
    for _ in range(n_customers):
        rand = random.random()
        if rand < 0.15:  # 15%は特別セグメント（休眠 or 高額のどちらか）
            customer_types.append(special_segment)
        elif rand < 0.25:  # 10%は優良顧客候補
            customer_types.append("loyal")
        elif rand < 0.35:  # 10%は離反リスク
            customer_types.append("churn_risk")
        else:  # 65%は通常パターン
            customer_types.append("normal")

    # 顧客タイプに応じた購入回数を設定
    purchase_counts = []
    for ctype in customer_types:
        if ctype == "dormant":
            # 休眠顧客: 1-2回のみ購入
            purchase_counts.append(random.randint(1, 2))
        elif ctype == "high_value":
            # 高額購入顧客: 1-2回のみ購入（金額は後で高く設定）
            purchase_counts.append(random.randint(1, 2))
        elif ctype == "loyal":
            # 優良顧客: 高頻度（15-40回）
            purchase_counts.append(random.randint(15, 40))
        elif ctype == "churn_risk":
            # 離反リスク: 中〜高頻度（5-20回）
            purchase_counts.append(random.randint(5, 20))
        else:
            # 通常パターン: 従来の分布
            rand = random.random()
            if rand < 0.40:
                purchase_counts.append(1)
            elif rand < 0.65:
                purchase_counts.append(2)
            elif rand < 0.85:
                purchase_counts.append(random.randint(3, 5))
            else:
                purchase_counts.append(random.randint(6, 15))

    # 合計がn_ordersになるように調整
    total = sum(purchase_counts)
    if total < n_orders:
        # 不足分をヘビーユーザー（loyal, churn_risk）に追加
        diff = n_orders - total
        heavy_users = [i for i, (c, t) in enumerate(zip(purchase_counts, customer_types))
                       if t in ("loyal", "churn_risk", "normal") and c >= 5]
        if heavy_users:
            for _ in range(diff):
                idx = random.choice(heavy_users)
                purchase_counts[idx] += 1
    elif total > n_orders:
        # 超過分を削減（dormant, high_valueは維持）
        while sum(purchase_counts) > n_orders:
            candidates = [i for i, (c, t) in enumerate(zip(purchase_counts, customer_types))
                          if c > 1 and t not in ("dormant", "high_value")]
            if candidates:
                idx = random.choice(candidates)
                purchase_counts[idx] -= 1
            else:
                break

    # 顧客IDと購入回数・タイプのマッピング
    customer_purchase_map = dict(zip(customer_ids, purchase_counts))
    customer_type_map = dict(zip(customer_ids, customer_types))

    # 選択された特別セグメントをログ出力
    special_segment_name = "休眠顧客" if special_segment == "dormant" else "高額購入顧客"
    print(f"      特別セグメント: {special_segment_name}")

    orders = []

    # 日付フォーマットのバリエーション
    date_formats = [
        "%Y-%m-%d",           # 2024-01-15
        "%Y/%m/%d",           # 2024/01/15
        "%d-%m-%Y",           # 15-01-2024
        "%Y年%m月%d日",        # 2024年01月15日
        "%m/%d/%Y",           # 01/15/2024
    ]

    order_counter = 0
    for customer_id, n_purchases in customer_purchase_map.items():
        ctype = customer_type_map[customer_id]

        # 顧客タイプに応じた基準購入日を設定
        # 分析日は2026年2月4日を想定。すべての日付は過去に設定する。
        if ctype == "dormant":
            # 休眠顧客: 2022年前半の古い日付（最終購入が約4年前）
            base_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 180))
        elif ctype == "high_value":
            # 高額購入顧客: 2026年1月（分析日直前、recency約30日）
            base_date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 30))
        elif ctype == "loyal":
            if special_segment == "high_value":
                # 高額購入顧客モード: loyalは2023年開始（低頻度顧客より古くなるよう）
                base_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 90))
            else:
                # 休眠顧客モード: loyalは2024年開始
                base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 90))
        elif ctype == "churn_risk":
            # 離反リスク: 2023年前半
            base_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 90))
        else:
            # 通常パターン（選択されたセグメントと購入回数に応じて調整）
            if special_segment == "dormant" and n_purchases <= 2:
                # 休眠顧客が選択 + 低頻度の場合：dormantと同じ古い日付範囲
                base_date = datetime(2022, 1, 1) + timedelta(days=random.randint(0, 180))
            elif special_segment == "dormant":
                # 休眠顧客が選択 + 中〜高頻度の場合：2024年開始
                base_date = datetime(2024, 1, 1) + timedelta(days=random.randint(0, 180))
            elif special_segment == "high_value" and n_purchases <= 2:
                # 高額購入顧客が選択 + 低頻度の場合：2026年1月（最も最近）
                base_date = datetime(2026, 1, 1) + timedelta(days=random.randint(0, 30))
            else:
                # 高額購入顧客が選択 + 中〜高頻度の場合：2023年
                base_date = datetime(2023, 1, 1) + timedelta(days=random.randint(0, 300))

        for purchase_num in range(n_purchases):
            order_counter += 1
            order_id = f"O{str(order_counter).zfill(6)}"
            product_id = random.choice(product_ids)

            # 注文日生成（顧客タイプに応じた間隔）
            # 分析日は2026年2月4日。すべての日付がこれより前になるよう設定
            analysis_date = datetime(2026, 2, 4)
            max_date = analysis_date - timedelta(days=7)  # 分析日の1週間前を上限

            if ctype == "loyal":
                order_date = base_date + timedelta(days=purchase_num * random.randint(5, 15))
            elif ctype == "churn_risk":
                order_date = base_date + timedelta(days=purchase_num * random.randint(7, 20))
            else:
                order_date = base_date + timedelta(days=purchase_num * random.randint(7, 30))

            # 日付が上限を超えたら上限に収める
            if order_date > max_date:
                order_date = max_date - timedelta(days=random.randint(0, 30))

            # 汚れ1: 日付フォーマットの不統一（約20%）
            if random.random() < 0.20:
                date_format = random.choice(date_formats)
                order_date_str = order_date.strftime(date_format)
            else:
                order_date_str = order_date.strftime("%Y-%m-%d")

            # 汚れ2: 日付が欠損（約3%）
            if random.random() < 0.03:
                order_date_str = None

            # 数量生成
            quantity = random.randint(1, 5)

            # 汚れ3: 異常な数量（約2%）
            if random.random() < 0.02:
                quantity = random.choice([-1, 0, 999, None])

            # 金額計算
            base_price = product_prices.get(product_id, 1000)

            # 高額購入顧客: 1回の購入が高額（5万〜30万円）
            if ctype == "high_value":
                total_amount = random.randint(50000, 300000)
                quantity = random.randint(5, 30)
            # 優良顧客: やや高めの購入額
            elif ctype == "loyal":
                total_amount = base_price * random.randint(2, 8)
                quantity = random.randint(2, 8)
            # 汚れ4: 極端な外れ値（約1%で100万円超の注文）
            # ただし、休眠顧客モードで低頻度顧客の場合は外れ値を生成しない
            # （高額購入顧客が発生するのを防ぐため）
            elif random.random() < 0.01 and not (special_segment == "dormant" and n_purchases <= 2):
                total_amount = random.randint(1000000, 5000000)
                quantity = random.randint(100, 500)
            else:
                total_amount = base_price * (quantity if isinstance(quantity, int) and quantity > 0 else 1)

            # 汚れ5: 金額が欠損または不正（約2%）
            if random.random() < 0.02:
                total_amount = random.choice([None, -500, 0])

            # 支払い方法
            payment_methods = ["クレジットカード", "銀行振込", "代金引換", "コンビニ払い", "PayPay",
                              "クレカ", "credit card", None]  # 表記揺れと欠損を含む
            payment_method = random.choices(payment_methods,
                                            weights=[0.35, 0.15, 0.1, 0.1, 0.15, 0.05, 0.05, 0.05])[0]

            # 配送ステータス
            statuses = ["delivered", "shipped", "pending", "cancelled", "返品", "配送済", None]
            status = random.choices(statuses, weights=[0.5, 0.15, 0.1, 0.1, 0.05, 0.05, 0.05])[0]

            orders.append({
                "order_id": order_id,
                "customer_id": customer_id,
                "product_id": product_id,
                "order_date": order_date_str,
                "quantity": quantity,
                "total_amount": total_amount,
                "payment_method": payment_method,
                "status": status,
            })

    return pd.DataFrame(orders)


def merge_and_save(customers_df: pd.DataFrame, products_df: pd.DataFrame,
                   orders_df: pd.DataFrame, output_path: str) -> pd.DataFrame:
    """
    3つのデータを結合してraw_data.csvとして保存
    """

    # 注文データを軸に顧客・商品情報を結合
    merged_df = orders_df.merge(customers_df, on="customer_id", how="left")
    merged_df = merged_df.merge(products_df, on="product_id", how="left")

    # カラム順序を整理
    column_order = [
        "order_id", "order_date", "customer_id", "customer_name", "email", "phone",
        "prefecture", "age", "gender", "product_id", "product_name", "category",
        "price", "quantity", "total_amount", "payment_method", "status"
    ]

    # 存在するカラムのみ選択
    merged_df = merged_df[[col for col in column_order if col in merged_df.columns]]

    # CSV保存
    merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return merged_df


def generate_data_summary(df: pd.DataFrame) -> dict:
    """
    生成したデータの汚れサマリーを出力
    """
    summary = {
        "total_records": len(df),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_emails": df["email"].duplicated().sum(),
        "date_format_issues": 0,
        "outlier_amounts": 0,
        "negative_values": 0,
    }

    # 日付フォーマットの問題をカウント
    standard_format = r"^\d{4}-\d{2}-\d{2}$"
    date_mask = df["order_date"].notna()
    summary["date_format_issues"] = len(df[date_mask]) - df.loc[date_mask, "order_date"].str.match(standard_format).sum()

    # 外れ値（100万円以上）をカウント
    amount_mask = df["total_amount"].notna()
    summary["outlier_amounts"] = (df.loc[amount_mask, "total_amount"] >= 1000000).sum()

    # 負の値をカウント
    for col in ["quantity", "total_amount"]:
        if col in df.columns:
            numeric_mask = pd.to_numeric(df[col], errors="coerce").notna()
            summary["negative_values"] += (pd.to_numeric(df.loc[numeric_mask, col], errors="coerce") < 0).sum()

    return summary


def generate_template_csv() -> pd.DataFrame:
    """
    分析に必要なカラム構成を持つテンプレートCSVを生成

    Returns:
        pd.DataFrame: サンプル1-2行を含むテンプレート
    """
    # 必須カラムとサンプルデータ
    template_data = {
        "order_id": ["O000001", "O000002"],
        "order_date": ["2024-01-15", "2024-02-20"],
        "customer_id": ["C00001", "C00002"],
        "customer_name": ["山田 太郎", "鈴木 花子"],
        "email": ["yamada@example.com", "suzuki@example.com"],
        "phone": ["090-1234-5678", "080-9876-5432"],
        "prefecture": ["東京都", "大阪府"],
        "age": [35, 28],
        "gender": ["M", "F"],
        "product_id": ["P0001", "P0002"],
        "product_name": ["洗剤", "シャンプー"],
        "category": ["日用品", "日用品"],
        "price": [500, 800],
        "quantity": [2, 1],
        "total_amount": [1000, 800],
        "payment_method": ["クレジットカード", "PayPay"],
        "status": ["delivered", "delivered"],
    }

    return pd.DataFrame(template_data)


def get_required_columns() -> list:
    """
    分析に必須のカラムリストを返す

    Returns:
        list: 必須カラム名のリスト
    """
    return [
        "order_id",
        "order_date",
        "customer_id",
        "product_id",
        "category",
        "total_amount",
    ]


def get_optional_columns() -> list:
    """
    分析でオプションとして使用するカラムリストを返す

    Returns:
        list: オプションカラム名のリスト
    """
    return [
        "customer_name",
        "email",
        "phone",
        "prefecture",
        "age",
        "gender",
        "product_name",
        "price",
        "quantity",
        "payment_method",
        "status",
    ]


def main():
    """
    メイン実行関数
    """
    print("=" * 60)
    print("ECサイト モックデータ生成スクリプト")
    print("=" * 60)

    # 出力ディレクトリ
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "raw_data.csv")

    # データ生成
    print("\n[1/4] 顧客データを生成中...")
    customers_df = generate_customers(n_customers=200)
    print(f"      -> {len(customers_df)} 件の顧客データを生成")

    print("\n[2/4] 商品データを生成中...")
    products_df = generate_products(n_products=50)
    print(f"      -> {len(products_df)} 件の商品データを生成")

    print("\n[3/4] 注文データを生成中...")
    orders_df = generate_orders(customers_df, products_df, n_orders=2000)
    print(f"      -> {len(orders_df)} 件の注文データを生成")

    print("\n[4/4] データを結合して保存中...")
    merged_df = merge_and_save(customers_df, products_df, orders_df, output_path)
    print(f"      -> {output_path} に保存完了")

    # サマリー出力
    print("\n" + "=" * 60)
    print("データ品質サマリー（意図的に含めた汚れ）")
    print("=" * 60)

    summary = generate_data_summary(merged_df)

    print(f"\n総レコード数: {summary['total_records']}")
    print(f"\n【欠損値】")
    for col, count in summary["null_counts"].items():
        if count > 0:
            print(f"  - {col}: {count} 件")

    print(f"\n【重複メールアドレス】: {summary['duplicate_emails']} 件")
    print(f"【日付フォーマット不統一】: {summary['date_format_issues']} 件")
    print(f"【外れ値（100万円以上）】: {summary['outlier_amounts']} 件")
    print(f"【負の値】: {summary['negative_values']} 件")

    print("\n" + "=" * 60)
    print("生成完了!")
    print("=" * 60)

    return merged_df


if __name__ == "__main__":
    main()
