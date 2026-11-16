from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
# 引入富途/牛牛 OpenD SDK (需确保已通过 pip install moomoo-api 安装)
from futu import OpenQuoteContext, RET_OK

app = FastAPI(title="Moomoo OpenD Bridge for Cycle Terminal")

# 允许跨域，方便网页前端直接请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 映射本地代码到富途标准代码格式
SYMBOL_MAP = {
    "SOXL": "US.SOXL",
    "SOXS": "US.SOXS",
    "TQQQ": "US.TQQQ",
    "SQQQ": "US.SQQQ",
    "QQQ": "US.QQQ",
    "SPY": "US.SPY",
    "SOXX": "US.SOXX"
}

@app.get("/api/quote/{symbol}")
def get_local_quote(symbol: symbol):
    target = SYMBOL_MAP.get(symbol.upper())
    if not target:
        raise HTTPException(status_code=400, detail="Unsupported symbol")
    
    try:
        # 连接本地 OpenD 网关 (默认端口 11111)
        quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
        ret, data = quote_ctx.get_market_snapshot([target])
        quote_ctx.close()
        
        if ret == RET_OK and not data.empty:
            row = data.iloc[0]
            # 提取现价、昨收、盘前/盘后价
            latest = float(row.get('last_price', 0))
            prev_close = float(row.get('prev_close_price', 0))
            high = float(row.get('high_price', latest))
            low = float(row.get('low_price', latest))
            volume = int(row.get('volume', 0))
            
            return {
                "symbol": symbol.upper(),
                "price": latest,
                "prev_close": prev_close,
                "high": high,
                "low": low,
                "volume": volume,
                "source": "moomoo_opend_local"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to fetch from OpenD")
    except Exception as e:
        # 若 OpenD 未启动或连接失败，抛出错误由前端降级回云端
        raise HTTPException(status_code=503, detail=f"OpenD connection error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8888)
