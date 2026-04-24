import streamlit as st
import pandas as pd
import time
import os
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION ---
st.set_page_config(
    page_title="EthicaAI Genesis Monitor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LANGUAGE & TEXT ---
LANG = {
    "KR": {
        "title": "🧬 EthicaAI Genesis: 자율 진화 모니터",
        "subtitle": "인공지능 에이전트 사회의 도덕적 진화 과정을 실시간으로 관찰합니다.",
        "sidebar_title": "설정 (Settings)",
        "refresh_rate": "새로고침 주기 (초)",
        "current_status": "현재 상태 (Current Status)",
        "gen": "세대 (Generation)",
        "coop": "협력률 (Cooperation Rate)",
        "best_coop": "최고 협력률 (Best Record)",
        "mode": "실험 모드 (Mode)",
        "param_search": "파라미터 탐색 (Parameter Search)",
        "rationale": "🤔 인공지능의 고민 (Theorist's Rationale)",
        "verdict": "👮 심판의 판정 (Critic's Verdict)",
        "success": "목표 달성! (Success)",
        "fail": "목표 미달 (Failure)",
        "chart_coop": "📉 협력률 변화 추이",
        "chart_param": "🧪 파라미터 탐색 공간 (Beta vs Alpha)",
        "desc_beta": "개입 강도 (Beta)",
        "desc_alpha": "민감도 (Alpha)",
        "wait": "데이터를 기다리는 중... (Waiting for data...)",
        "history_tab": "진화 기록 (History Log)",
        "data_tab": "상세 데이터 (Raw Data)"
    },
    "EN": {
        "title": "🧬 EthicaAI Genesis: Autonomous Evolution Monitor",
        "subtitle": "Real-time observation of moral evolution in multi-agent societies.",
        "sidebar_title": "Settings",
        "refresh_rate": "Refresh Rate (sec)",
        "current_status": "Current Status",
        "gen": "Generation",
        "coop": "Coop Rate",
        "best_coop": "Best Record",
        "mode": "Experiment Mode",
        "param_search": "Parameter Search",
        "rationale": "🤔 Theorist's Rationale",
        "verdict": "👮 Critic's Verdict",
        "success": "Goal Reached! (Success)",
        "fail": "Goal Missed (Failure)",
        "chart_coop": "📉 Cooperation Rate Trend",
        "chart_param": "🧪 Parameter Search Space (Beta vs Alpha)",
        "desc_beta": "Intervention (Beta)",
        "desc_alpha": "Sensitivity (Alpha)",
        "wait": "Waiting for data...",
        "history_tab": "History Log",
        "data_tab": "Raw Data"
    }
}

# Sidebar for Language
with st.sidebar:
    st.title("EthicaAI Genesis")
    lang_code = st.radio("Language / 언어", ["KR", "EN"], index=0)
    T = LANG[lang_code]
    
    st.divider()
    refresh_rate = st.slider(T["refresh_rate"], 1, 60, 5)
    
    # 연구 의제 상태 표시
    agenda_path = "experiments/evolution/research_agenda.json"
    if os.path.exists(agenda_path):
        import json as _json
        try:
            with open(agenda_path, "r", encoding="utf-8") as _f:
                _agenda = _json.load(_f)
            _questions = _agenda.get("questions", {})
            _total = len(_questions)
            _completed = sum(1 for q in _questions.values() if q["status"] == "completed")
            _failed = sum(1 for q in _questions.values() if q["status"] == "failed")
            _active = [q for q in _questions.values() if q["status"] == "active"]
            
            st.divider()
            st.subheader("🏛️ 연구소 현황" if lang_code == "KR" else "🏛️ Lab Status")
            if _total > 0:
                st.progress(_completed / _total, text=f"{_completed}/{_total} 완료")
            st.metric("완료" if lang_code == "KR" else "Done", _completed)
            st.metric("실패" if lang_code == "KR" else "Failed", _failed)
            st.metric("총 세대" if lang_code == "KR" else "Total Gen", _agenda.get("total_generations_run", 0))
            
            if _active:
                _aq = _active[0]
                st.info(f"📋 **{_aq['id']}**\n{_aq['question']}")
        except Exception:
            pass
    
    st.divider()
    st.info("""
    **EthicaAI v2.0**
    - **Goal**: Autonomous R&D
    - **Method**: SA-PPO + Mediator
    - **Engine**: Gemini 2.0 + JAX
    """)

    # v2.0: GPU/CPU 플랫폼 상태
    try:
        import jax
        _backend = jax.default_backend()
        _icon = "🚀" if _backend == "gpu" else "🐢"
        st.metric("Platform", f"{_icon} {_backend.upper()}")
    except Exception:
        st.metric("Platform", "❓ Unknown")

    # v2.0: 트리 탐색 상태
    _tree_path = "experiments/evolution/search_tree.json"
    if os.path.exists(_tree_path):
        import json as _json2
        try:
            with open(_tree_path, "r", encoding="utf-8") as _tf:
                _tree = _json2.load(_tf)
            st.divider()
            st.subheader("🌳 탐색 트리" if lang_code == "KR" else "🌳 Search Tree")
            st.metric("최고 CR" if lang_code == "KR" else "Best CR", f"{_tree.get('best_cr', 0):.4f}")
            st.metric("노드 수" if lang_code == "KR" else "Nodes", len(_tree.get("nodes", {})))
        except Exception:
            pass

# Main Content
st.title(T["title"])
st.markdown(f"*{T['subtitle']}*")

csv_path = "experiments/evolution/evolution_progress.csv"
history_path = "experiments/evolution/history.json"

def load_data():
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    
    try:
        data = []
        with open(csv_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 7:
                    continue  # 불량 데이터 건너뜀
                
                # 기본 7개 컬럼 매핑
                row = {
                    "Generation": parts[0],
                    "Beta": parts[1],
                    "Alpha": parts[2],
                    "Mode": parts[3],
                    "Coop_Prosocial": parts[4],
                    "Coop_Individualist": parts[5],
                    "Success": parts[6]
                }
                
                # 8번째 컬럼 (QuestionID) 처리
                if len(parts) >= 8:
                    row["QuestionID"] = parts[7]
                else:
                    row["QuestionID"] = None
                
                data.append(row)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # 숫자형 변환
        numeric_cols = ["Generation", "Beta", "Alpha", "Coop_Prosocial", "Coop_Individualist"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df
        
    except Exception as e:
        st.error(f"Data Load Error: {e}")
        return pd.DataFrame()

# Auto-refresh loop
placeholder = st.empty()

while True:
    df = load_data()
    
    with placeholder.container():
        if not df.empty and "Coop_Prosocial" in df.columns:
            try:
                # 1. KPI Metrics
                last_run = df.iloc[-1]
                best_run = df.loc[df["Coop_Prosocial"].idxmax()]
                
                current_coop = last_run['Coop_Prosocial']
                delta_color = "normal"
                if current_coop > 0.5:
                    delta_color = "inverse"
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric(T["gen"], int(last_run["Generation"]), delta=1)
                col2.metric(T["coop"], f"{current_coop:.4f}", delta=f"{current_coop - 0.1338:.4f}", delta_color=delta_color)
                col3.metric(T["mode"], last_run["Mode"])
                col4.metric(T["best_coop"], f"{best_run['Coop_Prosocial']:.4f}", f"Gen {int(best_run['Generation'])}")
                
                st.divider()

                # 2. Charts
                tab1, tab2, tab3 = st.tabs(["📊 " + T["chart_coop"], "🔍 " + T["chart_param"], "🌳 " + ("탐색 트리" if lang_code == "KR" else "Tree Search")])
                
                with tab1:
                    fig_coop = go.Figure()
                    fig_coop.add_trace(go.Scatter(x=df["Generation"], y=df["Coop_Prosocial"], mode='lines+markers', name='Prosocial', line=dict(color='#00CC96', width=3)))
                    fig_coop.add_trace(go.Scatter(x=df["Generation"], y=df["Coop_Individualist"], mode='lines', name='Individualist', line=dict(color='#EF553B', dash='dot')))
                    fig_coop.add_hline(y=0.5, line_dash="dash", line_color="green", annotation_text="Target (0.5)")
                    fig_coop.add_hline(y=0.1338, line_dash="dash", line_color="gray", annotation_text="Baseline (Nash)")
                    fig_coop.update_layout(title=T["chart_coop"], height=400, hovermode="x unified")
                    st.plotly_chart(fig_coop, use_container_width=True, key=f"main_chart_{int(time.time())}")

                with tab2:
                    fig_param = px.scatter(
                        df, x="Beta", y="Alpha",
                        color="Coop_Prosocial", size="Coop_Prosocial",
                        hover_data=["Generation", "Mode"],
                        labels={"Beta": T["desc_beta"], "Alpha": T["desc_alpha"]},
                        title=T["chart_param"],
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig_param, use_container_width=True, key=f"param_chart_{int(time.time())}")

                with tab3:
                    # v2.0: 트리 탐색 시각화
                    import json as _json3
                    _tree_path2 = "experiments/evolution/search_tree.json"
                    if os.path.exists(_tree_path2):
                        with open(_tree_path2, "r", encoding="utf-8") as _f3:
                            _tree2 = _json3.load(_f3)
                        
                        _nodes = _tree2.get("nodes", {})
                        if _nodes:
                            _cols = st.columns([1, 1])
                            _cols[0].metric("최고 CR" if lang_code == "KR" else "Best CR", f"{_tree2.get('best_cr', 0):.4f}")
                            _cols[1].metric("탐색 노드" if lang_code == "KR" else "Nodes", len(_nodes))
                            
                            _rows = []
                            for _nid, _nd in _nodes.items():
                                _cr = _nd.get("result", {}).get("cooperation_rate", "") if _nd.get("result") else ""
                                _rows.append({
                                    "ID": _nid,
                                    "Hypothesis": _nd.get("hypothesis", "")[:60],
                                    "Status": _nd.get("status", ""),
                                    "CR": _cr,
                                    "Parent": _nd.get("parent", "-"),
                                    "Children": len(_nd.get("children", [])),
                                })
                            st.dataframe(pd.DataFrame(_rows), use_container_width=True)
                        else:
                            st.info("트리 탐색이 아직 시작되지 않았습니다." if lang_code == "KR" else "Tree search not started yet.")
                    else:
                        st.info("트리 탐색 데이터가 없습니다." if lang_code == "KR" else "No tree search data.")

                # 3. Thinking Process (The Brain)
                st.subheader(T["history_tab"])
                
                if os.path.exists(history_path):
                    import json
                    try:
                        with open(history_path, "r", encoding="utf-8") as f:
                            history = json.load(f)
                        
                        for i, item in enumerate(reversed(history[-3:])):
                            gen_num = item.get('config', {}).get('GENESIS_GENERATION', '?')
                            ts = item.get('timestamp', '').split('T')[1][:8]
                            
                            rationale_en = item.get('config', {}).get('rationale', "No rationale.")
                            rationale_kr = item.get('config', {}).get('rationale_kr', "")
                            
                            if lang_code == "KR" and rationale_kr:
                                rationale = rationale_kr
                            else:
                                rationale = rationale_en
                            
                            success = item.get("success", False)
                            mode = item.get('config', {}).get('GENESIS_LOGIC_MODE', 'Unknown')
                            beta = item.get('config', {}).get('GENESIS_BETA', 0)
                            
                            with st.expander(f"🧬 Gen {gen_num} | {mode} (Beta={beta}) | {ts}", expanded=(i==0)):
                                st.markdown(f"**{T['rationale']}**")
                                st.info(rationale)
                                
                                # v2.0: 다차원 지표 표시
                                _si = item.get("stability_index")
                                _platform = item.get("platform", "")
                                _ia = item.get('config', {}).get('USE_INEQUITY_AVERSION', False)
                                if _si is not None:
                                    _mcols = st.columns(3)
                                    _mcols[0].metric("안정성" if lang_code == "KR" else "Stability", f"{_si:.4f}")
                                    _mcols[1].metric("IA", "✅" if _ia else "❌")
                                    _mcols[2].metric("Platform", _platform.upper() if _platform else "?")
                                
                                if success:
                                    st.success(f"🎉 {T['success']}")
                                else:
                                    st.error(f"❌ {T['fail']} (Coop: {item.get('result', {}).get('Prosocial', {}).get('cooperation_rate', 0.0):.4f})")
                    except Exception as e:
                        st.error(f"History Load Error: {e}")
                
                # 4. Raw Data Expander
                with st.expander(T["data_tab"]):
                    st.dataframe(df.sort_values("Generation", ascending=False), use_container_width=True)
            
            except Exception as e:
                st.error(f"Dashboard Error: {e}")
        else:
            st.warning(T["wait"])
            
    time.sleep(refresh_rate)
