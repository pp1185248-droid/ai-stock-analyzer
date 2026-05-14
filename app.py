"""
╔══════════════════════════════════════════════════════════════════╗
║     AI-POWERED STOCK ANALYSIS WEB APP - Full Functional Script   ║
║     Tech Stack: Streamlit + yfinance + Plotly + Gemini AI        ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google import genai
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# YOUR PERSONAL GEMINI API KEY (Pre-configured)
# ══════════════════════════════════════════════════════════
DEFAULT_GEMINI_KEY = "AIzaSyBHO30Vpn1FHUcwFihXzPIKN8xYHxxS0iM"

# ══════════════════════════════════════════════════════════
# STREAMLIT PAGE CONFIG
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Stock Analyzer Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .buy-signal {
        background: linear-gradient(135deg, #00501e, #006b29);
        border: 2px solid #00ff7f;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #00ff7f;
        margin-bottom: 15px;
    }
    .sell-signal {
        background: linear-gradient(135deg, #500000, #6b0000);
        border: 2px solid #ff4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ff4444;
        margin-bottom: 15px;
    }
    .hold-signal {
        background: linear-gradient(135deg, #2a2a00, #3d3d00);
        border: 2px solid #ffd700;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #ffd700;
        margin-bottom: 15px;
    }
    .ai-box {
        background: linear-gradient(135deg, #0d0d2b, #1a1a4e);
        border: 1px solid #4444ff;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════
# SECTION 1: DATA FETCHING
# ══════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def fetch_stock_data(ticker: str, period: str = "6mo", interval: str = "1d"):
    """yfinance se live OHLCV data fetch karo (new multi-level column format)"""
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            return None, None

        # Flatten multi-level columns (new yfinance format)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.reset_index(inplace=True)
        # Rename Datetime to Date if needed
        if 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'Date'}, inplace=True)

        df = df.dropna(subset=['Close', 'Open', 'High', 'Low'])

        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except:
            info = {}

        return df, info
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None, None


# ══════════════════════════════════════════════════════════
# SECTION 2: TECHNICAL ANALYSIS ENGINE
# ══════════════════════════════════════════════════════════
class TechnicalAnalysis:

    @staticmethod
    def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
        delta = data.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(data: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def calculate_ema(data: pd.Series, period: int) -> pd.Series:
        return data.ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_bollinger_bands(data: pd.Series, period=20, std_dev=2.0):
        middle = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        width = (upper - lower) / middle * 100
        return upper, middle, lower, width

    @staticmethod
    def calculate_atr(high, low, close, period=14) -> pd.Series:
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    @classmethod
    def add_all_indicators(cls, df: pd.DataFrame) -> pd.DataFrame:
        close = df['Close']
        high  = df['High']
        low   = df['Low']

        df['RSI']  = cls.calculate_rsi(close)
        df['MACD'], df['MACD_Signal'], df['MACD_Hist'] = cls.calculate_macd(close)
        df['EMA_9']   = cls.calculate_ema(close, 9)
        df['EMA_20']  = cls.calculate_ema(close, 20)
        df['EMA_50']  = cls.calculate_ema(close, 50)
        df['EMA_200'] = cls.calculate_ema(close, 200)
        df['BB_Upper'], df['BB_Middle'], df['BB_Lower'], df['BB_Width'] = \
            cls.calculate_bollinger_bands(close)
        df['ATR'] = cls.calculate_atr(high, low, close)
        df['Volume_SMA']   = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        return df


# ══════════════════════════════════════════════════════════
# SECTION 3: PATTERN DETECTION ENGINE
# ══════════════════════════════════════════════════════════
class PatternDetector:

    @staticmethod
    def find_pivots(df: pd.DataFrame, window: int = 10):
        highs, lows = [], []
        for i in range(window, len(df) - window):
            if df['High'].iloc[i] == df['High'].iloc[i-window:i+window+1].max():
                highs.append((i, float(df['High'].iloc[i])))
            if df['Low'].iloc[i] == df['Low'].iloc[i-window:i+window+1].min():
                lows.append((i, float(df['Low'].iloc[i])))
        return highs, lows

    @staticmethod
    def find_support_resistance(df: pd.DataFrame, lookback=50):
        recent = df.tail(lookback)
        current_price = float(df['Close'].iloc[-1])
        supports, resistances = [], []

        for i in range(2, len(recent) - 2):
            h = float(recent['High'].iloc[i])
            l = float(recent['Low'].iloc[i])
            if (h > recent['High'].iloc[i-1] and h > recent['High'].iloc[i-2] and
                    h > recent['High'].iloc[i+1] and h > recent['High'].iloc[i+2]):
                resistances.append(h)
            if (l < recent['Low'].iloc[i-1] and l < recent['Low'].iloc[i-2] and
                    l < recent['Low'].iloc[i+1] and l < recent['Low'].iloc[i+2]):
                supports.append(l)

        support_levels    = sorted([s for s in supports    if s < current_price], reverse=True)[:3]
        resistance_levels = sorted([r for r in resistances if r > current_price])[:3]
        return {'support': support_levels, 'resistance': resistance_levels, 'current_price': current_price}

    @staticmethod
    def detect_double_top(pivot_highs, tolerance=0.03):
        patterns = []
        for i in range(len(pivot_highs) - 1):
            for j in range(i + 1, len(pivot_highs)):
                h1, h2 = pivot_highs[i][1], pivot_highs[j][1]
                if abs(h1 - h2) / h1 < tolerance:
                    gap = pivot_highs[j][0] - pivot_highs[i][0]
                    if 10 <= gap <= 60:
                        patterns.append({
                            'type': 'Double Top', 'signal': 'BEARISH',
                            'description': f'Double Top ~{(h1+h2)/2:.2f} par. Bearish reversal!'
                        })
        return patterns[-2:] if patterns else []

    @staticmethod
    def detect_double_bottom(pivot_lows, tolerance=0.03):
        patterns = []
        for i in range(len(pivot_lows) - 1):
            for j in range(i + 1, len(pivot_lows)):
                l1, l2 = pivot_lows[i][1], pivot_lows[j][1]
                if abs(l1 - l2) / l1 < tolerance:
                    gap = pivot_lows[j][0] - pivot_lows[i][0]
                    if 10 <= gap <= 60:
                        patterns.append({
                            'type': 'Double Bottom', 'signal': 'BULLISH',
                            'description': f'Double Bottom ~{(l1+l2)/2:.2f} par. Bullish reversal!'
                        })
        return patterns[-2:] if patterns else []

    @staticmethod
    def detect_bullish_flag(df: pd.DataFrame):
        recent = df.tail(30)
        first_half  = recent.head(15)
        second_half = recent.tail(15)
        pole_move = (float(first_half['Close'].iloc[-1]) - float(first_half['Close'].iloc[0])) / float(first_half['Close'].iloc[0]) * 100
        flag_move = (float(second_half['Close'].iloc[-1]) - float(second_half['Close'].iloc[0])) / float(second_half['Close'].iloc[0]) * 100
        if pole_move > 8 and -5 < flag_move < 0:
            return {'type': 'Bullish Flag', 'signal': 'BULLISH',
                    'description': f'Bullish Flag: {pole_move:.1f}% pole + consolidation. Breakout expected!'}
        return {}

    @staticmethod
    def detect_head_shoulders(pivot_highs):
        for i in range(len(pivot_highs) - 2):
            ls, head, rs = pivot_highs[i][1], pivot_highs[i+1][1], pivot_highs[i+2][1]
            if head > ls and head > rs and abs(ls - rs) / ls < 0.05:
                return {'type': 'Head & Shoulders', 'signal': 'BEARISH',
                        'description': f'H&S pattern: Head at {head:.2f}. Strong bearish reversal!'}
        return {}

    @classmethod
    def detect_all_patterns(cls, df: pd.DataFrame):
        pivot_highs, pivot_lows = cls.find_pivots(df)
        sr = cls.find_support_resistance(df)
        patterns = {
            'double_top':         cls.detect_double_top(pivot_highs),
            'double_bottom':      cls.detect_double_bottom(pivot_lows),
            'bullish_flag':       cls.detect_bullish_flag(df),
            'head_shoulders':     cls.detect_head_shoulders(pivot_highs),
            'support_resistance': sr
        }
        return patterns, pivot_highs, pivot_lows


# ══════════════════════════════════════════════════════════
# SECTION 4: SIGNAL GENERATION (SCORING SYSTEM)
# ══════════════════════════════════════════════════════════
class SignalGenerator:

    @staticmethod
    def generate_signals(df: pd.DataFrame, patterns: dict) -> dict:
        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        score  = 0
        signals = []

        # RSI
        rsi = float(latest['RSI'])
        if rsi < 30:
            score += 2; signals.append(f"✅ RSI={rsi:.1f} — Oversold zone (Strong Buy)")
        elif rsi < 45:
            score += 1; signals.append(f"✅ RSI={rsi:.1f} — Bullish territory")
        elif rsi > 70:
            score -= 2; signals.append(f"❌ RSI={rsi:.1f} — Overbought zone (Strong Sell)")
        elif rsi > 55:
            score -= 1; signals.append(f"❌ RSI={rsi:.1f} — Bearish territory")
        else:
            signals.append(f"⚪ RSI={rsi:.1f} — Neutral")

        # MACD
        macd, macd_sig = float(latest['MACD']), float(latest['MACD_Signal'])
        if macd > macd_sig and float(prev['MACD']) <= float(prev['MACD_Signal']):
            score += 2; signals.append("✅ MACD Bullish Crossover! Strong buy signal")
        elif macd > macd_sig:
            score += 1; signals.append("✅ MACD above signal — Bullish momentum")
        elif macd < macd_sig and float(prev['MACD']) >= float(prev['MACD_Signal']):
            score -= 2; signals.append("❌ MACD Bearish Crossover! Strong sell signal")
        elif macd < macd_sig:
            score -= 1; signals.append("❌ MACD below signal — Bearish momentum")

        # EMA
        ema50, ema200, close = float(latest['EMA_50']), float(latest['EMA_200']), float(latest['Close'])
        if ema50 > ema200:
            score += 1; signals.append(f"✅ Golden Cross: EMA50 ({ema50:.2f}) > EMA200 ({ema200:.2f})")
        else:
            score -= 1; signals.append(f"❌ Death Cross: EMA50 ({ema50:.2f}) < EMA200 ({ema200:.2f})")
        if close > ema50:
            score += 1; signals.append(f"✅ Price above EMA50 — Strong position")
        else:
            score -= 1; signals.append(f"❌ Price below EMA50 — Weak position")

        # Bollinger Bands
        bb_upper, bb_lower = float(latest['BB_Upper']), float(latest['BB_Lower'])
        bb_pos = (close - bb_lower) / (bb_upper - bb_lower) * 100
        if bb_pos < 10:
            score += 2; signals.append(f"✅ Price near Lower BB ({bb_pos:.0f}%) — Potential bounce")
        elif bb_pos > 90:
            score -= 2; signals.append(f"❌ Price near Upper BB ({bb_pos:.0f}%) — Potential pullback")
        else:
            signals.append(f"⚪ BB Position: {bb_pos:.0f}% — Mid range")

        # Volume
        vol_ratio = float(latest['Volume_Ratio'])
        if vol_ratio > 1.5:
            if close > float(prev['Close']):
                score += 1; signals.append(f"✅ High Volume ({vol_ratio:.1f}x) on up day — Institutional buying!")
            else:
                score -= 1; signals.append(f"❌ High Volume ({vol_ratio:.1f}x) on down day — Selling pressure!")

        # Patterns
        if patterns.get('bullish_flag'):
            score += 2; signals.append(f"✅ {patterns['bullish_flag']['description']}")
        if patterns.get('double_bottom'):
            score += 2; signals.append(f"✅ {patterns['double_bottom'][-1]['description']}")
        if patterns.get('double_top'):
            score -= 2; signals.append(f"❌ {patterns['double_top'][-1]['description']}")
        if patterns.get('head_shoulders'):
            score -= 2; signals.append(f"❌ {patterns['head_shoulders']['description']}")

        # ATR-based targets
        atr = float(latest['ATR'])
        sr  = patterns.get('support_resistance', {})

        if score >= 4:
            final_signal = "🟢 STRONG BUY";  signal_class = "buy-signal"
            stop_loss = close - 2*atr; t1 = close + 2*atr; t2 = close + 4*atr
        elif score >= 2:
            final_signal = "🟡 BUY";         signal_class = "buy-signal"
            stop_loss = close - 1.5*atr; t1 = close + 1.5*atr; t2 = close + 3*atr
        elif score <= -4:
            final_signal = "🔴 STRONG SELL"; signal_class = "sell-signal"
            stop_loss = close + 2*atr; t1 = close - 2*atr; t2 = close - 4*atr
        elif score <= -2:
            final_signal = "🟠 SELL";        signal_class = "sell-signal"
            stop_loss = close + 1.5*atr; t1 = close - 1.5*atr; t2 = close - 3*atr
        else:
            final_signal = "⚪ HOLD/NEUTRAL"; signal_class = "hold-signal"
            stop_loss = close - atr; t1 = close + atr; t2 = close + 2*atr

        return {
            'final_signal': final_signal, 'signal_class': signal_class,
            'score': score, 'signals': signals,
            'stop_loss': stop_loss, 'target_1': t1, 'target_2': t2,
            'current_price': close, 'rsi': rsi, 'macd': macd, 'atr': atr,
            'ema50': ema50, 'ema200': ema200, 'bb_width': float(latest['BB_Width'])
        }


# ══════════════════════════════════════════════════════════
# SECTION 5: AI REASONING (GEMINI)
# ══════════════════════════════════════════════════════════
def get_ai_analysis(ticker, signal_data, df, patterns, api_key) -> str:
    try:
        client = genai.Client(api_key=api_key)
        sr = patterns.get('support_resistance', {})
        found_patterns = [p for p in ['double_top','double_bottom','bullish_flag','head_shoulders']
                          if patterns.get(p)]

        prompt = f"""
You are an expert Indian stock market analyst. Analyze this stock in Hinglish (mix of Hindi and English).

Stock: {ticker}
Current Price: {signal_data['current_price']:.2f}
Signal: {signal_data['final_signal']}
Score: {signal_data['score']}/10

Technical Indicators:
- RSI(14): {signal_data['rsi']:.1f}
- MACD: {signal_data['macd']:.4f}
- EMA 50: {signal_data['ema50']:.2f}
- EMA 200: {signal_data['ema200']:.2f}
- ATR: {signal_data['atr']:.2f}
- BB Width: {signal_data['bb_width']:.2f}%

Chart Patterns: {found_patterns if found_patterns else 'None'}
Support Levels: {sr.get('support', [])}
Resistance Levels: {sr.get('resistance', [])}
Stop Loss: {signal_data['stop_loss']:.2f}
Target 1: {signal_data['target_1']:.2f}
Target 2: {signal_data['target_2']:.2f}

Individual signals:
{chr(10).join(signal_data['signals'])}

Please provide (in Hinglish):
1. Is signal kyun aaya — clear explanation
2. Key risks kya hain
3. Trader ko kya watch karna chahiye
4. Risk management suggestion (position sizing, stop loss advice)

Keep it 4-5 paragraphs, practical, conversational.
"""
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ AI Analysis unavailable: {str(e)}\n\nSidebar mein valid Gemini API key daalo."


# ══════════════════════════════════════════════════════════
# SECTION 6: INTERACTIVE CHART (PLOTLY)
# ══════════════════════════════════════════════════════════
def create_main_chart(df, ticker, patterns, pivot_highs, pivot_lows):
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} — Price Chart', 'Volume', 'RSI (14)', 'MACD'),
        row_heights=[0.5, 0.15, 0.15, 0.2]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price',
        increasing_line_color='#00ff7f', decreasing_line_color='#ff4444',
        increasing_fillcolor='#00ff7f', decreasing_fillcolor='#ff4444'
    ), row=1, col=1)

    # EMAs
    for period, color in [(20, '#FFD700'), (50, '#00BFFF'), (200, '#FF69B4')]:
        fig.add_trace(go.Scatter(
            x=df['Date'], y=df[f'EMA_{period}'], name=f'EMA {period}',
            line=dict(color=color, width=1.5)
        ), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Upper'], name='BB Upper',
        line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Lower'], name='BB Lower',
        line=dict(color='rgba(255,165,0,0.5)', width=1, dash='dash'),
        fill='tonexty', fillcolor='rgba(255,165,0,0.05)'), row=1, col=1)

    # Support / Resistance
    sr = patterns.get('support_resistance', {})
    for lvl in sr.get('support', []):
        fig.add_hline(y=lvl, line_dash='dot', line_color='#00FF7F', opacity=0.6,
                      annotation_text=f'S: {lvl:.1f}', annotation_font_color='#00FF7F', row=1, col=1)
    for lvl in sr.get('resistance', []):
        fig.add_hline(y=lvl, line_dash='dot', line_color='#FF4444', opacity=0.6,
                      annotation_text=f'R: {lvl:.1f}', annotation_font_color='#FF4444', row=1, col=1)

    # Pivot markers
    if pivot_highs:
        ph_x = [df['Date'].iloc[p[0]] for p in pivot_highs if p[0] < len(df)]
        ph_y = [p[1] for p in pivot_highs if p[0] < len(df)]
        fig.add_trace(go.Scatter(x=ph_x, y=ph_y, mode='markers', name='Pivot High',
            marker=dict(symbol='triangle-down', size=9, color='#FF4444')), row=1, col=1)
    if pivot_lows:
        pl_x = [df['Date'].iloc[p[0]] for p in pivot_lows if p[0] < len(df)]
        pl_y = [p[1] for p in pivot_lows if p[0] < len(df)]
        fig.add_trace(go.Scatter(x=pl_x, y=pl_y, mode='markers', name='Pivot Low',
            marker=dict(symbol='triangle-up', size=9, color='#00FF7F')), row=1, col=1)

    # Volume
    vol_colors = ['#00ff7f' if c >= o else '#ff4444'
                  for c, o in zip(df['Close'], df['Open'])]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], name='Volume',
        marker_color=vol_colors, opacity=0.7), row=2, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Volume_SMA'], name='Vol SMA',
        line=dict(color='#FFD700', width=1.5)), row=2, col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI',
        line=dict(color='#9B59B6', width=2)), row=3, col=1)
    fig.add_hline(y=70, line_dash='dash', line_color='#FF4444', opacity=0.7, row=3, col=1)
    fig.add_hline(y=30, line_dash='dash', line_color='#00FF7F', opacity=0.7, row=3, col=1)
    fig.add_hrect(y0=70, y1=100, fillcolor='red',   opacity=0.06, row=3, col=1)
    fig.add_hrect(y0=0,  y1=30,  fillcolor='green', opacity=0.06, row=3, col=1)

    # MACD
    macd_colors = ['#00ff7f' if v >= 0 else '#ff4444' for v in df['MACD_Hist']]
    fig.add_trace(go.Bar(x=df['Date'], y=df['MACD_Hist'], name='Histogram',
        marker_color=macd_colors, opacity=0.7), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD'], name='MACD',
        line=dict(color='#00BFFF', width=1.5)), row=4, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MACD_Signal'], name='Signal',
        line=dict(color='#FF69B4', width=1.5)), row=4, col=1)

    fig.update_layout(
        template='plotly_dark', paper_bgcolor='#0e1117', plot_bgcolor='#0e1117',
        height=900, showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='right', x=1, font=dict(size=10)),
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color='#ffffff')
    )
    fig.update_xaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)',
                     showspikes=True, spikecolor='white', spikethickness=1)
    fig.update_yaxes(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
    return fig


# ══════════════════════════════════════════════════════════
# SECTION 7: MAIN STREAMLIT APP
# ══════════════════════════════════════════════════════════
def main():

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("# 📈 AI Stock Analyzer Pro")
        st.markdown("---")

        st.markdown("### 🔍 Stock Selection")
        ticker = st.text_input(
            "Stock Ticker",
            value="TCS.NS",
            help="NSE: RELIANCE.NS | BSE: RELIANCE.BO | US: AAPL"
        ).upper().strip()

        st.markdown("**Quick Select:**")
        quick = {
            "TCS": "TCS.NS", "RELIANCE": "RELIANCE.NS",
            "NIFTY50": "^NSEI", "SENSEX": "^BSESN",
            "HDFC Bk": "HDFCBANK.NS", "INFY": "INFY.NS",
            "AAPL": "AAPL", "TSLA": "TSLA"
        }
        cols = st.columns(2)
        for i, (name, sym) in enumerate(quick.items()):
            if cols[i % 2].button(name, key=f"q_{name}", use_container_width=True):
                ticker = sym

        st.markdown("---")
        st.markdown("### ⏰ Time & Interval")
        period = st.selectbox("Period",
            ["1mo","3mo","6mo","1y","2y"], index=2,
            format_func=lambda x: {"1mo":"1 Month","3mo":"3 Months","6mo":"6 Months","1y":"1 Year","2y":"2 Years"}[x])
        interval = st.selectbox("Candle",
            ["1d","1wk"], index=0,
            format_func=lambda x: {"1d":"Daily","1wk":"Weekly"}[x])

        st.markdown("---")
        st.markdown("### 🤖 AI Settings")
        gemini_key = st.text_input("Gemini API Key", type="password",
            value=DEFAULT_GEMINI_KEY,
            placeholder="AIzaSy...", help="Free key: aistudio.google.com")
        enable_ai = st.toggle("Enable AI Analysis", value=True)

        st.markdown("---")
        analyze_btn = st.button("🚀 ANALYZE STOCK", use_container_width=True, type="primary")

        st.markdown("---")
        st.caption("⚠️ Sirf educational purpose ke liye. Investment decisions apni research aur financial advisor ki salaah par base karo.")

    # ── Header ─────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center; padding:10px 0 20px 0;'>
        <h1 style='color:#00ff7f; font-size:2.5em; margin:0;'>📊 AI Stock Analysis Terminal</h1>
        <p style='color:#888;'>NSE • BSE • Global Markets | Technical + AI Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Run Analysis ───────────────────────────────────────
    if analyze_btn:
        st.session_state['ticker']   = ticker
        st.session_state['period']   = period
        st.session_state['interval'] = interval
        st.session_state['analyzed'] = True

    if st.session_state.get('analyzed'):
        ticker   = st.session_state.get('ticker', ticker)
        period   = st.session_state.get('period', period)
        interval = st.session_state.get('interval', interval)

        with st.spinner(f"🔄 {ticker} ka live data fetch ho raha hai..."):
            df, stock_info = fetch_stock_data(ticker, period, interval)

        if df is None or len(df) < 50:
            st.error(f"❌ '{ticker}' ka data nahi mila. Ticker dobara check karo.")
            st.info("💡 NSE: RELIANCE.NS | BSE: RELIANCE.BO | Index: ^NSEI | US: AAPL")
            st.session_state['analyzed'] = False
            return

        with st.spinner("🔢 Technical indicators calculate ho rahe hain..."):
            df = TechnicalAnalysis.add_all_indicators(df)

        with st.spinner("🔍 Chart patterns scan ho rahe hain..."):
            patterns, pivot_highs, pivot_lows = PatternDetector.detect_all_patterns(df)

        signal_data = SignalGenerator.generate_signals(df, patterns)

        latest = df.iloc[-1]
        prev   = df.iloc[-2]
        price_change = float(latest['Close']) - float(prev['Close'])
        price_change_pct = price_change / float(prev['Close']) * 100
        company_name = (stock_info or {}).get('longName', ticker)
        currency = "₹" if ('.NS' in ticker or '.BO' in ticker) else "$"

        # Company header
        date_str = str(latest['Date'])[:10]
        st.markdown(f"""
        <div style='background:linear-gradient(135deg,#1a1a2e,#16213e);
                    border:1px solid #0f3460; border-radius:10px; padding:15px 25px; margin-bottom:20px;'>
            <h2 style='color:#fff; margin:0;'>{company_name}</h2>
            <span style='color:#888;'>{ticker} | Last Updated: {date_str}</span>
        </div>
        """, unsafe_allow_html=True)

        # Metrics row
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("💰 Price",
                  f"{currency}{signal_data['current_price']:.2f}",
                  f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
        c2.metric("📊 RSI (14)", f"{signal_data['rsi']:.1f}",
                  "Oversold ✅" if signal_data['rsi'] < 30 else ("Overbought ⚠️" if signal_data['rsi'] > 70 else "Normal"), delta_color="off")
        c3.metric("📈 EMA 50", f"{signal_data['ema50']:.2f}",
                  "Price Above ✅" if signal_data['current_price'] > signal_data['ema50'] else "Price Below ❌", delta_color="off")
        c4.metric("📉 EMA 200", f"{signal_data['ema200']:.2f}",
                  "Golden Cross ✅" if signal_data['ema50'] > signal_data['ema200'] else "Death Cross ❌", delta_color="off")
        c5.metric("🌊 ATR", f"{signal_data['atr']:.2f}",
                  f"Vol {float(latest['Volume_Ratio']):.1f}x avg", delta_color="off")

        st.markdown("---")

        # Signal + Details
        st.markdown("## 🎯 Trading Signal")
        sig_col, det_col = st.columns([1, 2])

        with sig_col:
            st.markdown(f"""
            <div class='{signal_data["signal_class"]}'>
                {signal_data['final_signal']}<br>
                <small>Score: {signal_data['score']}/10</small>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### 🎯 Price Targets")
            cp  = signal_data['current_price']
            t1  = signal_data['target_1']
            t2  = signal_data['target_2']
            sl  = signal_data['stop_loss']
            st.markdown(f"""
            <div style='background:#1a1a2e;border-radius:8px;padding:10px;margin:4px 0;'>
                <span style='color:#888'>Current:</span>
                <b style='color:#fff'>{currency}{cp:.2f}</b>
            </div>
            <div style='background:#1a2a1a;border:1px solid #00ff7f33;border-radius:8px;padding:10px;margin:4px 0;'>
                <span style='color:#888'>Target 1:</span>
                <b style='color:#00ff7f'>{currency}{t1:.2f}</b>
                <span style='color:#888;font-size:.8em'> ({(t1/cp-1)*100:+.1f}%)</span>
            </div>
            <div style='background:#1a2a1a;border:1px solid #00ff7f55;border-radius:8px;padding:10px;margin:4px 0;'>
                <span style='color:#888'>Target 2:</span>
                <b style='color:#00ff7f'>{currency}{t2:.2f}</b>
                <span style='color:#888;font-size:.8em'> ({(t2/cp-1)*100:+.1f}%)</span>
            </div>
            <div style='background:#2a1a1a;border:1px solid #ff444433;border-radius:8px;padding:10px;margin:4px 0;'>
                <span style='color:#888'>Stop Loss:</span>
                <b style='color:#ff4444'>{currency}{sl:.2f}</b>
                <span style='color:#888;font-size:.8em'> ({(sl/cp-1)*100:+.1f}%)</span>
            </div>
            """, unsafe_allow_html=True)

        with det_col:
            st.markdown("### 📋 Signal Breakdown")
            for sig in signal_data['signals']:
                st.markdown(f"- {sig}")

            st.markdown("### 🔭 Detected Patterns")
            found = []
            if patterns.get('bullish_flag'):
                found.append(f"✅ **Bullish Flag** — {patterns['bullish_flag']['description']}")
            if patterns.get('double_bottom'):
                found.append(f"✅ **Double Bottom** — {patterns['double_bottom'][-1]['description']}")
            if patterns.get('double_top'):
                found.append(f"❌ **Double Top** — {patterns['double_top'][-1]['description']}")
            if patterns.get('head_shoulders'):
                found.append(f"❌ **{patterns['head_shoulders']['type']}** — {patterns['head_shoulders']['description']}")
            if found:
                for p in found: st.markdown(p)
            else:
                st.info("Current timeframe mein koi significant pattern nahi mila.")

            sr = patterns.get('support_resistance', {})
            if sr.get('support') or sr.get('resistance'):
                sc, rc = st.columns(2)
                with sc:
                    st.markdown("**🟢 Support:**")
                    for l in sr.get('support', []): st.markdown(f"  - {currency}{l:.2f}")
                with rc:
                    st.markdown("**🔴 Resistance:**")
                    for l in sr.get('resistance', []): st.markdown(f"  - {currency}{l:.2f}")

        st.markdown("---")

        # Chart
        st.markdown("## 📊 Interactive Chart")
        with st.spinner("Chart render ho raha hai..."):
            fig = create_main_chart(df, ticker, patterns, pivot_highs, pivot_lows)
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # AI Analysis
        st.markdown("## 🤖 AI Analysis (Hinglish)")
        if enable_ai and gemini_key:
            with st.spinner("🧠 Gemini AI analysis generate kar raha hai..."):
                ai_text = get_ai_analysis(ticker, signal_data, df, patterns, gemini_key)
            st.markdown(f"""
            <div class='ai-box'>
                <h4 style='color:#6677ff;margin-top:0;'>🤖 AI Analyst Report</h4>
                {ai_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("🤖 AI Analysis ke liye sidebar mein **Gemini API Key** daalo (free: aistudio.google.com).")

        # Raw data
        with st.expander("📋 Historical Data (Last 30 Days)"):
            show_df = df[['Date','Open','High','Low','Close','Volume','RSI','MACD','EMA_50','EMA_200']].tail(30).copy()
            show_df = show_df.round(2)
            show_df['Date'] = show_df['Date'].astype(str).str[:10]
            st.dataframe(show_df, use_container_width=True, hide_index=True)

    else:
        # Welcome screen
        st.markdown("""
        <div style='text-align:center; padding:50px 20px;'>
            <div style='font-size:80px;'>📈</div>
            <h2 style='color:#00ff7f;'>AI Stock Analyzer Pro mein Swagat Hai!</h2>
            <p style='color:#888; font-size:1.1em; max-width:600px; margin:0 auto;'>
                Left sidebar se koi bhi stock enter karo aur
                <b style='color:#00ff7f;'>🚀 ANALYZE STOCK</b> dabao.
            </p>
            <div style='margin-top:40px; display:flex; justify-content:center; gap:20px; flex-wrap:wrap;'>
                <div style='background:#1a1a2e;border:1px solid #0f3460;border-radius:10px;padding:20px;width:160px;'>
                    <div style='font-size:2em;'>📊</div>
                    <div style='color:#fff;margin-top:8px;font-weight:bold;'>Technical Analysis</div>
                    <div style='color:#888;font-size:.8em;margin-top:4px;'>RSI, MACD, Bollinger, EMA</div>
                </div>
                <div style='background:#1a1a2e;border:1px solid #0f3460;border-radius:10px;padding:20px;width:160px;'>
                    <div style='font-size:2em;'>🔭</div>
                    <div style='color:#fff;margin-top:8px;font-weight:bold;'>Pattern Detection</div>
                    <div style='color:#888;font-size:.8em;margin-top:4px;'>Double Top/Bottom, Flag, H&S</div>
                </div>
                <div style='background:#1a1a2e;border:1px solid #0f3460;border-radius:10px;padding:20px;width:160px;'>
                    <div style='font-size:2em;'>🤖</div>
                    <div style='color:#fff;margin-top:8px;font-weight:bold;'>AI Reasoning</div>
                    <div style='color:#888;font-size:.8em;margin-top:4px;'>Gemini AI — Hinglish explanation</div>
                </div>
                <div style='background:#1a1a2e;border:1px solid #0f3460;border-radius:10px;padding:20px;width:160px;'>
                    <div style='font-size:2em;'>🎯</div>
                    <div style='color:#fff;margin-top:8px;font-weight:bold;'>Smart Signals</div>
                    <div style='color:#888;font-size:.8em;margin-top:4px;'>Buy/Sell + Target + Stop Loss</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
