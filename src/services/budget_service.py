import logging

logger = logging.getLogger("BudgetGuard")

class BudgetGuard:
    def __init__(self, limit_yen=50.0):
        self.limit_yen = limit_yen
        self.current_cost = 0.0
        
        # 100万トークンあたりの単価目安 (円) - $1=150円換算
        # 参考: https://ai.google.dev/gemini-api/docs/pricing
        
        # Flash / Flash-Lite: 
        # Input $0.075 (~11.25円) / Output $0.30 (~45円) 
        # ※以前の画像($0.10/$0.40)より安くなっていますが、安全側に倒して少し高めに設定するか、正確に合わせるか。
        # ここでは安全マージン込みで $0.10 / $0.40 (15円 / 60円) を維持しつつ、Proを修正します。
        
        # Pro: 
        # Input $1.25 (~187.5円) / Output $10.00 (~1500円)
        # ※<=128kコンテキストの場合。これを超えると倍額になりますが、基本はこちらを使用。
        
        self.rates = {
            # Flash系 (Lite含む)
            "gemini-2.5-flash-lite": {"input": 15.0,  "output": 60.0},
            "gemini-2.0-flash":      {"input": 15.0,  "output": 60.0},
            "gemini-2.5-flash":      {"input": 15.0,  "output": 60.0}, 
            "gemini-1.5-flash":      {"input": 15.0,  "output": 60.0},
            
            # Pro / High-Intelligence系 (1.5, 2.0, 2.5, 3.0)
            "gemini-2.5-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-2.0-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-1.5-pro":        {"input": 187.5, "output": 1500.0},
            "gemini-3":              {"input": 300.0, "output": 1800.0}, # Gemini 3は仮の高め設定
        }

    def check_and_record(self, model_name: str, input_chars: int, output_chars: int):
        """コストを計算し、累積する。上限を超えたら例外を投げる。"""
        rate = None
        # 部分一致でレートを探す (例: "models/gemini-1.5-pro-latest" -> "gemini-1.5-pro")
        for key in self.rates:
            if key in model_name:
                rate = self.rates[key]
                break
        
        if not rate:
            # 安全策: "pro" が名前に含まれていたらPro価格、それ以外はFlash価格を適用
            if "pro" in model_name.lower():
                rate = self.rates["gemini-1.5-pro"]
            else:
                rate = self.rates["gemini-2.5-flash-lite"]

        input_cost = (input_chars / 1_000_000) * rate["input"]
        output_cost = (output_chars / 1_000_000) * rate["output"]
        total_cost = input_cost + output_cost
        
        self.current_cost += total_cost
        
        logger.info(f"💰 Cost: +{total_cost:.4f}円 (Total: {self.current_cost:.2f} / {self.limit_yen}円) [{model_name}]")

        if self.current_cost > self.limit_yen:
            logger.error("💸 BUDGET EXCEEDED! Stopping execution to save money.")
            raise Exception(f"Budget Limit Exceeded: Used {self.current_cost:.2f}JPY (Limit: {self.limit_yen}JPY)")