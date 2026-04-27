import re
import requests

def is_time_query(msg):
    msg = msg.lower()
    return msg.startswith("saat kaç") or "tarih" in msg

def is_currency_query(msg):
    msg = msg.lower()
    return ("dolar" in msg) and ("çevr" in msg or "çevir" in msg)

def foreign_currency(user_msg_lower):
    match1 = re.search(r"dolara çevir\s+([\d.,₺tl\s]+)", user_msg_lower)
    match2 = re.search(r"([\d.,]+)\s*(tl|₺)\s*(kaç\s*dolar)?", user_msg_lower)

    if match1 or match2:
        miktar_str = match1.group(1) if match1 else match2.group(1)
        miktar_temiz = re.sub(r"[^\d.,]", "", miktar_str).replace(",", ".")
        try:
            miktar = float(miktar_temiz)
        except ValueError:
            return "Geçerli bir sayı giriniz."
        try:
            response = requests.get("https://doviz.dev/v1/usd.json")
            response.raise_for_status()
            data = response.json()
            usd_try = float(data["USDTRY"])
            tl_to_usd = miktar / usd_try
            return f"{miktar} TL yaklaşık {tl_to_usd:.2f} USD eder."
        except Exception:
            return "Döviz kuru alınırken bir hata oluştu."
    return None



