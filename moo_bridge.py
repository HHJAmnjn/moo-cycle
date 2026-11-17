import os, time, json
from http.server import HTTPServer, BaseHTTPRequestHandler
import pandas as pd

try:
    from moomoo import OpenQuoteContext, SubType, RET_OK, SysConfig
except ImportError:
    from futu import OpenQuoteContext, SubType, RET_OK, SysConfig

SYMBOL_MAP = {
    "SOXL": "US.SOXL",
    "SOXS": "US.SOXS",
    "TQQQ": "US.TQQQ",
    "SQQQ": "US.SQQQ",
    "QQQ":  "US.QQQ",
    "SPY":  "US.SPY",
    "SOXX": "US.SOXX"
}
ALL_CODES = list(SYMBOL_MAP.values())
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(BASE_DIR, "live_price.json")

quote_ctx = None

def get_ctx():
    global quote_ctx
    if quote_ctx is None:
        SysConfig.enable_proto_encrypt(False)
        quote_ctx = OpenQuoteContext(host="127.0.0.1", port=11111)
        quote_ctx.subscribe(ALL_CODES, [SubType.QUOTE], subscribe_push=False)
    return quote_ctx

def extract_session_quote(row):
    bjt_h = time.localtime().tm_hour
    reg_p = float(row.get("last_price", 0) or 0)
    prev_c = float(row.get("prev_close_price", 0) or reg_p)

    if 8 <= bjt_h < 16:
        session_cols = [
            ("overnight_price", "overnight_high_price", "overnight_low_price", "overnight_volume", "overnight_turnover"),
            ("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover"),
            ("pre_price", "pre_high_price", "pre_low_price", "pre_volume", "pre_turnover")
        ]
    elif 16 <= bjt_h < 22:
        session_cols = [
            ("pre_price", "pre_high_price", "pre_low_price", "pre_volume", "pre_turnover"),
            ("overnight_price", "overnight_high_price", "overnight_low_price", "overnight_volume", "overnight_turnover")
        ]
    elif 5 <= bjt_h < 8:
        session_cols = [
            ("after_price", "after_high_price", "after_low_price", "after_volume", "after_turnover")
        ]
    else:
        session_cols = []

    active_p = reg_p
    hi = float(row.get("high_price", 0) or reg_p)
    lo = float(row.get("low_price", 0) or reg_p)
    vol = float(row.get("volume", 0) or 0)
    turn = float(row.get("turnover", 0) or 0)

    for p_col, h_col, l_col, v_col, t_col in session_cols:
        if p_col in row and pd.notna(row[p_col]) and float(row[p_col]) > 0:
            active_p = float(row[p_col])
            if h_col in row and pd.notna(row[h_col]) and float(row[h_col]) > 0:
                hi = float(row[h_col])
            if l_col in row and pd.notna(row[l_col]) and float(row[l_col]) > 0:
                lo = float(row[l_col])
            if v_col in row and t_col in row and pd.notna(row[v_col]) and pd.notna(row[t_col]) and float(row[v_col]) > 0:
                vol = float(row[v_col])
                turn = float(row[t_col])
            break

    vwap = round(turn / vol, 2) if (vol > 0 and turn > 0) else round((hi + lo + active_p) / 3.0, 2)
    return {
        "price": round(active_p, 2),
        "reg_close": round(reg_p if (8 <= bjt_h < 22 and reg_p > 0) else prev_c, 2),
        "high": round(max(hi, active_p), 2),
        "low": round(min(lo, active_p) if lo > 0 else active_p, 2),
        "vwap": vwap,
        "volume": int(vol)
    }

def fetch_opend_payload():
    global quote_ctx
    try:
        ctx = get_ctx()
        ret, data = ctx.get_market_snapshot(ALL_CODES)
        if ret != RET_OK or data.empty:
            raise RuntimeError("OpenD snapshot empty")
        res = {}
        for _, row in data.iterrows():
            sym = str(row["code"]).replace("US.", "")
            res[sym] = extract_session_quote(row)
        payload = {"ts": time.strftime("%H:%M:%S"), "source": "moomoo_opend", "quotes": res}
        try:
            with open(OUT_JSON + ".tmp", "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(OUT_JSON + ".tmp", OUT_JSON)
        except Exception:
            pass
        return payload
    except Exception as e:
        if quote_ctx:
            try: quote_ctx.close()
            except Exception: pass
            quote_ctx = None
        raise e

class BridgeHandler(BaseHTTPRequestHandler):
    def _send_cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors()
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path in ("/", "/index.html", "/dashboard.html"):
            try:
                with open(os.path.join(BASE_DIR, "index.html"), "rb") as f:
                    content = f.read()
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(content)
                return
            except Exception as e:
                pass

        if path == "/api/quotes_all":
            try:
                payload = fetch_opend_payload()
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                err = json.dumps({"error": str(e)}).encode("utf-8")
                self.send_response(503)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(err)
            return

        if path.startswith("/api/quote/"):
            sym = path.split("/api/quote/")[-1].upper()
            try:
                payload = fetch_opend_payload()
                if sym in payload["quotes"]:
                    out = {"symbol": sym, **payload["quotes"][sym], "source": "moomoo_opend"}
                    self.send_response(200)
                    self._send_cors()
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps(out).encode("utf-8"))
                    return
            except Exception as e:
                pass
            self.send_response(503)
            self._send_cors()
            self.end_headers()
            return

        self.send_response(404)
        self._send_cors()
        self.end_headers()

    def log_message(self, format, *args):
        return

if __name__ == "__main__":
    server = HTTPServer(("127.0.0.1", 8888), BridgeHandler)
    print("=================================================================")
    print("🚀 Moomoo OpenD 零依赖本地桥接服务已启动！(监听 127.0.0.1:8888)")
    print("👉 本地直连访问地址 : http://127.0.0.1:8888")
    print("👉 GitHub Pages 亦可通过 8888 端口自动读取本地 OpenD 实时行情")
    print("=================================================================")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 本地桥接服务已停止。")
