import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Sales Forecast", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #d8b4f2; }

div[data-testid="stSidebar"] { background: #1a1a1a; }
div[data-testid="stSidebar"] * { color: #999 !important; }
div[data-testid="stSidebar"] hr { border-color: #2e2e2e; margin: 16px 0; }

.headline {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 700;
    color: #1a1a1a;
    line-height: 1;
    letter-spacing: -1px;
    margin-bottom: 6px;
}
.subline { font-size: 14px; color: #888; font-weight: 300; margin-bottom: 20px; }

.hint-box {
    background: #efe9df;
    border-radius: 6px;
    padding: 14px 18px;
    font-size: 13px;
    color: #7a6e5f;
    line-height: 1.7;
    margin-bottom: 28px;
}

.stat-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 22px 18px;
    height: 100%;
}
.stat-top { height: 3px; border-radius: 3px; margin-bottom: 16px; }
.stat-label { font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #aaa; margin-bottom: 8px; }
.stat-num { font-size: 30px; font-weight: 600; color: #1a1a1a; line-height: 1; }
.stat-desc { font-size: 12px; color: #bbb; margin-top: 10px; line-height: 1.6; }

.box {
    background: #ffffff;
    border-radius: 8px;
    padding: 24px;
}
.box-title { font-size: 16px; font-weight: 600; color: #1a1a1a; margin-bottom: 2px; }
.box-sub { font-size: 12px; color: #aaa; margin-bottom: 18px; }

.action-card {
    background: #1a1a1a;
    border-radius: 8px;
    padding: 22px;
    height: 100%;
}
.ac-step { font-size: 9px; letter-spacing: 2px; text-transform: uppercase; color: #444; margin-bottom: 8px; }
.ac-title { font-size: 15px; font-weight: 600; color: #f8f6f1; margin-bottom: 10px; }
.ac-body { font-size: 13px; color: #666; line-height: 1.8; }
.ac-hi { color: #c8a46e; font-weight: 500; }

.section-label {
    font-size: 10px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #c8a46e;
    margin-bottom: 6px;
}
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    color: #1a1a1a;
    margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('train_small.csv', nrows=500000)
    df['date'] = pd.to_datetime(df['date'])
    df['Month'] = df['date'].dt.to_period('M')
    m = df.groupby('Month')['sales'].sum().reset_index()
    m['Month'] = m['Month'].dt.to_timestamp()
    m['Month_Num'] = np.arange(len(m))
    m['Month_of_Year'] = m['Month'].dt.month
    m['Year'] = m['Month'].dt.year
    return m

@st.cache_data
def load_family():
    df = pd.read_csv('train.csv', nrows=500000)
    df['date'] = pd.to_datetime(df['date'])
    f = df.groupby('family')['sales'].sum().reset_index()
    return f.sort_values('sales', ascending=True).tail(12)

def train(data):
    cols = ['Month_Num', 'Month_of_Year', 'Year']
    cut = int(len(data) * 0.8)
    tr, te = data[:cut], data[cut:]
    m = LinearRegression()
    m.fit(tr[cols], tr['sales'])
    p = m.predict(te[cols])
    return m, mean_absolute_error(te['sales'], p), te, p

def make_future(model, data, n):
    cols = ['Month_Num', 'Month_of_Year', 'Year']
    ln = data['Month_Num'].max()
    ld = data['Month'].max()
    rows = []
    for i in range(1, n+1):
        d = ld + pd.DateOffset(months=i)
        rows.append({'Month': d, 'Month_Num': ln+i, 'Month_of_Year': d.month, 'Year': d.year})
    out = pd.DataFrame(rows)
    out['forecast'] = model.predict(out[cols])
    std = data['sales'].std() * 0.12
    out['hi'] = out['forecast'] + std
    out['lo'] = (out['forecast'] - std).clip(lower=0)
    return out

CHART = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='#ffffff',
    font=dict(color='#aaa', family='DM Sans', size=12),
    margin=dict(l=0, r=0, t=10, b=0),
    hovermode='x unified'
)
GRID = dict(gridcolor='#f0ece6', zeroline=False, showline=True, linecolor='#ece8e2')


with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 20px'>
        <div style='font-family:Playfair Display,serif; font-size:18px; color:'#aca6d2'>Sales Forecast</div>
        <div style='font-size:10px; letter-spacing:2px; text-transform:uppercase;color:#7a9e7e;font-weight:700;margin-top:4px'>Ecuador Retail</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#7a9e7e;font-weight:700; margin-bottom:8px'>Forecast window</div>", unsafe_allow_html=True)
    n_months = st.slider("Months ahead", min_value=3, max_value=12, value=6, label_visibility="collapsed")
    st.markdown(f"<div style='font-size:24px; font-weight:600; color:#c8a46e; margin-top:4px'>{n_months} months</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:10px; letter-spacing:2px; text-transform:uppercase; color:#7a9e7e;font-weight:700; margin-bottom:12px'>What each section shows</div>", unsafe_allow_html=True)
    sections = [
        ("4 key numbers", "Quick snapshot of where things stand"),
        ("Big chart", "All past sales plus the forecast line"),
        ("Accuracy check", "How close the model was on hidden data"),
        ("Month table", "Exact numbers you can plan with"),
        ("What sells most", "Top product categories by volume"),
        ("Action plan", "Three things to do based on the forecast"),
    ]
    for title, desc in sections:
        st.markdown(f"""
        <div style='border-left:2px solid #c8a46e; padding:6px 10px; margin-bottom:8px'>
            <div style='font-size:12px;color:#7a9e7e;font-weight:700'>{title}</div>
            <div style='font-size:11px; color:#c8a46e; margin-top:2px; line-height:1.5'>{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#7a9e7e;font-weight:700;line-height:2.2'>
        Country — Ecuador<br>
        Stores — Retail chain<br>
        Period — 2013 to 2017<br>
        Records — 500,000 rows<br>
        Model — Linear Regression<br>
        Source — Kaggle
    </div>
    """, unsafe_allow_html=True)


with st.spinner("Loading data..."):
    data = load_data()
    fam = load_family()
    model, mae, te, preds = train(data)
    fut = make_future(model, data, n_months)

peak = fut.loc[fut['forecast'].idxmax()]
slow = fut.loc[fut['forecast'].idxmin()]
avg_f = fut['forecast'].mean()
total_f = fut['forecast'].sum()


st.markdown('<div class="headline">How are sales doing?</div>', unsafe_allow_html=True)
st.markdown(f'<div class="subline"><span style="color:#7a9e7e;font-weight:700">A complete picture of past performance and what to expect over the next {n_months} months — built on real retail data from Ecuador.</span></div>', unsafe_allow_html=True)

st.markdown("""
<div class="hint-box">
    The dark line shows real past sales. The terracotta dashed line is the model's prediction for the future.
    The shaded band shows the range where sales will most likely land.
    Use the slider on the left to look closer or further ahead.
</div>
""", unsafe_allow_html=True)


c1, c2, c3, c4 = st.columns(4, gap="small")
stat_data = [
    ("#c8a46e", "Total sold, all time", f"{data['sales'].sum()/1e6:.1f}M units", "Every item sold across all stores since records began."),
    ("#1a1a1a", "Typical month", f"{data['sales'].mean()/1e3:.0f}K units", "What an average month looks like across the whole chain."),
    ("#7a9e7e", f"Next {n_months} months", f"{total_f/1e6:.1f}M units", "What the model expects to be sold over your forecast window."),
    ("#c87a5a", "Prediction error", f"{mae/1e3:.0f}K units", "How far off the model was when tested on hidden data."),
]
for col, (color, label, num, desc) in zip([c1, c2, c3, c4], stat_data):
    with col:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-top" style="background:{color}"></div>
            <div class="stat-label"><strong>{label}</strong></div>
            <div class="stat-num">{num}</div>
            <div class="stat-desc"><span style="color:#7a9e7e">{desc}</span></div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<div class="section-label"><strong>The full picture</strong></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Sales history and what comes next</div>', unsafe_allow_html=True)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=data['Month'], y=data['sales'],
    name='Past sales',
    line=dict(color='#1a1a1a', width=2),
    fill='tozeroy', fillcolor='rgba(26,26,26,0.04)',
    hovertemplate='%{x|%b %Y} — %{y:,.0f} units<extra></extra>'
))
fig.add_trace(go.Scatter(
    x=pd.concat([fut['Month'], fut['Month'][::-1]]),
    y=pd.concat([fut['hi'], fut['lo'][::-1]]),
    fill='toself', fillcolor='rgba(200,164,110,0.15)',
    line=dict(color='rgba(0,0,0,0)'),
    name='Likely range', hoverinfo='skip'
))
fig.add_trace(go.Scatter(
    x=fut['Month'], y=fut['forecast'],
    name='Forecast',
    line=dict(color='#c87a5a', width=2.5, dash='dash'),
    mode='lines+markers',
    marker=dict(size=7, color='#c87a5a', line=dict(color='#f8f6f1', width=2)),
    hovertemplate='%{x|%b %Y} — %{y:,.0f} units (forecast)<extra></extra>'
))
fig.update_layout(
    **CHART,
    legend=dict(bgcolor='rgba(248,246,241,0.95)', bordercolor='#e8e2d8',
                borderwidth=1, font=dict(color='#7a0d10', size=12),
                orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
    xaxis=dict(**GRID, tickformat='%b %Y', tickfont=dict(color='#7a9e7e')),
    yaxis=dict(**GRID, tickformat=',d', tickfont=dict(color='#7a9e7e')),
    height=420
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


left, right = st.columns([1.2, 0.8], gap="small")

with left:
    st.markdown('<div class="section-label"><strong>Accuracy check</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Did the model get it right?</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; color:#7a9e7e; margin-bottom:16px; font-weight:700'>The last 20% of data was hidden from the model during training. This shows how close its predictions were to what actually happened.</div>", unsafe_allow_html=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=te['Month'], y=te['sales'], name='What actually happened',
        line=dict(color='#1a1a1a', width=2.5),
        hovertemplate='%{x|%b %Y} — %{y:,.0f}<extra></extra>'
    ))
    fig2.add_trace(go.Scatter(
        x=te['Month'], y=preds, name='What the model predicted',
        line=dict(color='#c87a5a', width=2, dash='dot'),
        hovertemplate='%{x|%b %Y} — %{y:,.0f}<extra></extra>'
    ))
    fig2.update_layout(
        **CHART,
        legend=dict(bgcolor='rgba(248,246,241,0.95)', bordercolor='#e8e2d8',
                    borderwidth=1, font=dict(color='#444', size=11),
                    orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0),
        xaxis=dict(**GRID, tickformat='%b %Y', tickfont=dict(color='#7a9e7e')),
        yaxis=dict(**GRID, tickformat=',d', tickfont=dict(color='#7a9e7e')),
        height=340
    )
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.markdown('<div class="section-label"><strong>Month by month</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Exact forecast numbers</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:13px; color:#7a9e7e; margin-bottom:16px; font-weight:700'>Low and high show the range where sales will most likely land each month.</div>", unsafe_allow_html=True)

    tbl = fut[['Month', 'forecast', 'lo', 'hi']].copy()
    tbl['Month'] = tbl['Month'].dt.strftime('%B %Y')
    tbl['forecast'] = tbl['forecast'].apply(lambda x: f"{x:,.0f}")
    tbl['lo'] = tbl['lo'].apply(lambda x: f"{x:,.0f}")
    tbl['hi'] = tbl['hi'].apply(lambda x: f"{x:,.0f}")
    tbl.columns = ['Month', 'Forecast', 'Low', 'High']
    st.dataframe(tbl, use_container_width=True, hide_index=True, height=340)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<div class="section-label"><strong>Product breakdown</strong></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">What sells the most</div>', unsafe_allow_html=True)
st.markdown("<div style='font-size:13px; color:#7a9e7e; margin-bottom:16px; font-weight:700'>Total units sold per product category across the entire dataset. Longer bar means more volume.</div>", unsafe_allow_html=True)

colors = ['#e8e2d8'] * len(fam)
colors[-1] = '#c8a46e'
colors[-2] = '#c8a46e'
colors[-3] = '#c8a46e'

fig3 = go.Figure(go.Bar(
    x=fam['sales'], y=fam['family'],
    orientation='h',
    marker=dict(color=colors),
    hovertemplate='%{y} — %{x:,.0f} units<extra></extra>'
))
fig3.update_layout(
    **CHART,
    xaxis=dict(**GRID, tickformat=',d', tickfont=dict(color='#7a9e7e')),
    yaxis=dict(gridcolor='rgba(0,0,0,0)', zeroline=False, tickfont=dict(color='#7a9e7e')),
    height=380
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<div class="section-label"><strong>What to do with this</strong></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">Action plan based on the forecast</div>', unsafe_allow_html=True)

a1, a2, a3 = st.columns(3, gap="small")

with a1:
    st.markdown(f"""
    <div class="action-card">
        <div class="ac-step" style="color:#7a9e7e;">Action 01</div>
        <div class="ac-title">Order stock early</div>
        <div class="ac-body">
            Peak demand hits in <span class="ac-hi">{peak['Month'].strftime('%B %Y')}</span>
            at around <span class="ac-hi">{peak['forecast']:,.0f} units</span>.
            Place supplier orders 3 to 4 weeks before so shelves are fully stocked when demand arrives.
        </div>
    </div>""", unsafe_allow_html=True)

with a2:
    st.markdown(f"""
    <div class="action-card">
        <div class="ac-step" style="color:#7a9e7e;">Action 02</div>
        <div class="ac-title">Run a promotion in the slow month</div>
        <div class="ac-body">
            Sales dip in <span class="ac-hi">{slow['Month'].strftime('%B %Y')}</span>
            to around <span class="ac-hi">{slow['forecast']:,.0f} units</span>.
            A targeted discount or bundle offer this month can pull customers in and offset the natural dip.
        </div>
    </div>""", unsafe_allow_html=True)

with a3:
    st.markdown(f"""
    <div class="action-card">
        <div class="ac-step" style="color:#7a9e7e;">Action 03</div>
        <div class="ac-title">Plan your budget now</div>
        <div class="ac-body">
            Monthly average over the next <span class="ac-hi">{n_months} months</span>
            is <span class="ac-hi">{avg_f:,.0f} units</span>.
            Use this as the baseline for supplier contracts, staffing levels and expense planning.
        </div>
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; color:#c8a46e; font-size:11px; letter-spacing:1.5px;
            text-transform:uppercase; border-top:1px solid #e8e2d8;
            padding-top:20px; margin-top:40px'>
    Future Interns ML Internship &nbsp;&nbsp; Store Sales Dataset, Kaggle &nbsp;·&nbsp; Linear Regression
</div>
""", unsafe_allow_html=True)