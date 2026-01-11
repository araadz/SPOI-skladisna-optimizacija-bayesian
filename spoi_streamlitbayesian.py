"""
================================================================================
SPOI WAREHOUSE OPTIMIZATION - STREAMLIT APP
================================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from datetime import datetime

# Import modela
import warehouse_model_bayesian as model

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="SPOI Optimizacija skladišnih pozicija",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #1E3A8A; text-align: center;}
    .sub-header {font-size: 1.2rem; color: #6B7280; text-align: center; margin-bottom: 2rem;}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<p class="main-header"> Optimizacija skladišta</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Optimizacija rasporeda artikala korištenjem ILP solvera</p>', unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Parametri")

    AUTO_TUNE = st.checkbox("Automatski optimizuj parametre")
    LOCKED = AUTO_TUNE

    if AUTO_TUNE:
        st.info("Parametri su zaključani jer je uključena automatska optimizacija.")

    COST_A = st.slider(
        "α - Horizontalna (H)",
        0.5, 5.0,
        2.0,
        0.5,
        disabled=LOCKED)

    COST_B = st.slider(
        "β - Vertikalna (V>2)",
        1.0, 10.0,
        5.0,
        0.5,
        disabled=LOCKED)

    COST_C = st.slider(
        "γ - Dubina (E)",
        0.5, 5.0,
        2.0,
        0.5,
        disabled=LOCKED)

    st.markdown("---")

    TAU = st.slider(
        "τ - Temperatura",
        2.0, 20.0,
        8.0,
        1.0,
        disabled=LOCKED)

    DEMAND_MULTIPLIER = st.slider(
        "λ - Multiplikator",
        5, 30,
        15,
        1,
        disabled=LOCKED)

    st.markdown("---")
    N_PICKS = st.number_input("Pickova", 100, 5000, 1000, 100)
    st.markdown("---")

    N_TRIALS = st.slider(
        "Broj iteracija (auto)",
        10, 30,
        10,
        disabled=not AUTO_TUNE)
    
    if not AUTO_TUNE:
        st.caption("Broj iteracija se definiše samo kod automatske optimizacije.")

    st.markdown("---")
    st.latex(r"C = \alpha H + \beta \max(0,V-2) + \gamma(4-E)")
    st.latex(r"U = e^{-C/\tau} \cdot (1 + \lambda \frac{d}{d_{max}})")


# Parametri dict
params = {
    'COST_A': COST_A, 'COST_B': COST_B, 'COST_C': COST_C,
    'TAU': TAU, 'DEMAND_MULTIPLIER': DEMAND_MULTIPLIER, 'N_PICKS': N_PICKS
}

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3 = st.tabs(["📂 Učitaj & Optimiziraj", "📊 Rezultati", "📈 Grafici"])

# ============================================================
# TAB 1
# ============================================================
with tab1:
    st.header(" Upload Excel Fajla")
    
    # Upute
    with st.expander(" Upute za pripremu fajla", expanded=False):
        st.markdown("""
        ### Obavezne kolone:
        | Kolona | Opis |
        |--------|------|
        | **H** | Horizontalna pozicija (1-15) |
        | **V** | Vertikalni nivo (1-5) |
        | **E** | Dubina u regalu (1-4) |
        | **izlaz** | Potražnja artikla |
        
        ### Opcionalno:
        | Kolona | Opis |
        |--------|------|
        | **TEZINA_KAT** | Kategorija težine (1-8) |
        """)
    
    # UPLOAD
    uploaded_file = st.file_uploader(
        " Odaberi Excel fajl (.xlsx)",
        type=['xlsx', 'xls'],
        help="Upload Excel sa kolonama: H, V, E, izlaz"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            
            # Validacija
            missing = [c for c in ['H', 'V', 'E', 'izlaz'] if c not in df.columns]
            if missing:
                st.error(f"Nedostaju kolone: {missing}")
                st.stop()
            
            # Pripremi podatke
            df = model.prepare_data(df)
            n = len(df)
            
            st.success(f" Učitan file **{n}** artikala")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("H", f"{int(df['H'].min())} - {int(df['H'].max())}")
            col2.metric("V", f"{int(df['V'].min())} - {int(df['V'].max())}")
            col3.metric("E", f"{int(df['E'].min())} - {int(df['E'].max())}")
            col4.metric("Izlaz", f"{int(df['izlaz'].min())} - {int(df['izlaz'].max())}")
            
            st.session_state['df_raw'] = df
            st.session_state['data_loaded'] = True
            
            with st.expander(" Pregled podataka"):
                st.dataframe(df.head(15), use_container_width=True)
                
        except Exception as e:
            st.error(f" Greška: {e}")
    
    st.markdown("---")
    st.header(" Optimizacija")
    
    if st.button(" OPTIMIZIRAJ", type="primary", use_container_width=True):
        if 'data_loaded' not in st.session_state:
            st.warning("Prvo uploadaj Excel fajl!")
            st.stop()
    
        progress = st.progress(0)
        status = st.empty()
    
        status.text("Pokretanje optimizacije")
        progress.progress(10)
    
        try:
            if AUTO_TUNE:
                st.subheader("Optimizacija parametara")
    
                progress_bar = st.progress(0)
                iter_text = st.empty()
                best_text = st.empty()
    
                def progress_callback(i, total, best_score):
                    progress_bar.progress(i / total)
                    iter_text.markdown(f"**Iteracija:** {i} / {total}")
                    best_text.markdown(f"**Najbolji utility do sada:** {best_score:.3f}")
    
                with st.spinner("Optimiziram parametre..."):
                    best_params, best_score = model.bayesian_optimize_parameters(
                        st.session_state['df_raw'],
                        n_trials=N_TRIALS,
                        progress_callback=progress_callback
                    )
    
                params = best_params
                st.success("Optimizacija parametara završena")
    
            status.text("Pokrećem ILP optimizaciju rasporeda...")
            progress.progress(70)
    
            results = model.optimize(st.session_state['df_raw'], params)
    
            st.session_state['results'] = results
            st.session_state['optimized'] = True
    
            progress.progress(100)
            status.text("Optimizacija završena")
    
            st.success("Gotovo! Idi na tabove Rezultati i Grafici")
    
        except Exception as e:
            st.error(f"Greška: {e}")


# ============================================================
# TAB 2
# ============================================================
with tab2:
    st.header("Rezultati")
    
    if 'optimized' not in st.session_state:
        st.warning("Prvo pokreni optimizaciju!")
    else:
        r = st.session_state['results']
        
        # Metrike
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Utility", f"{r['opt_utils'].sum():.2f}", f"+{r['improvement']:.1f}%")
        col2.metric("Sim. Cost", f"{r['opt_sim']:,.0f}", f"-{r['cost_reduction']:.1f}%", delta_color="inverse")
        col3.metric("Weighted H", f"{r['opt_wH']:.2f}", f"-{r['h_reduction']:.1f}%", delta_color="inverse")
        col4.metric("Weighted V", f"{r['opt_wV']:.2f}", f"-{r['v_reduction']:.1f}%", delta_color="inverse")
        
        st.markdown("---")

        st.subheader("Korišteni parametri")

        
        if 'AUTO_TUNE' in globals() and AUTO_TUNE:
            hp = r['params']
            mode = "Automatski optimizovani"
        else:
            hp = params
            mode = "Ručno zadani (slideri)"
        
        hp_df = pd.DataFrame({
            'Parametar': [
                'α (COST_A)',
                'β (COST_B)',
                'γ (COST_C)',
                'τ (TAU)',
                'λ (DEMAND_MULTIPLIER)',
                'N_PICKS'
            ],
            'Vrijednost': [
                f"{hp['COST_A']:.2f}",
                f"{hp['COST_B']:.2f}",
                f"{hp['COST_C']:.2f}",
                f"{hp['TAU']:.2f}",
                f"{hp['DEMAND_MULTIPLIER']}",
                f"{hp['N_PICKS']}"
            ]
        })
        
        st.caption(f"Način odabira parametara: **{mode}**")
        st.dataframe(hp_df, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Tabela
        st.subheader("Usporedba")
        comp = pd.DataFrame({
            'Metrika': ['Total Utility', 'Simulation Cost', 'Weighted H', 'Weighted V', 'Premješteno'],
            'Početno': [f"{r['init_utils'].sum():.2f}", f"{r['init_sim']:,.0f}", 
                       f"{r['init_wH']:.2f}", f"{r['init_wV']:.2f}", "-"],
            'Optimizirano': [f"{r['opt_utils'].sum():.2f}", f"{r['opt_sim']:,.0f}",
                            f"{r['opt_wH']:.2f}", f"{r['opt_wV']:.2f}", f"{r['moved']}"],
            'Promjena': [f"+{r['improvement']:.2f}%", f"-{r['cost_reduction']:.2f}%",
                        f"-{r['h_reduction']:.2f}%", f"-{r['v_reduction']:.2f}%", 
                        f"{r['moved']/r['n_items']*100:.1f}%"]
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Top 10
        st.subheader("Top 10 Promjena")
        df = r['df']
        df_pos = r['df_positions']
        opt = r['opt_assign']
        
        top10 = []
        for idx in df.nlargest(10, 'izlaz').index:
            j = opt[idx]
            naziv = df.iloc[idx].get('Naziv artikla', f'#{idx}')
            if pd.isna(naziv): naziv = f'#{idx}'
            top10.append({
                'Naziv': str(naziv)[:25],
                'Potražnja': int(df.iloc[idx]['izlaz']),
                'H': f"{int(df.iloc[idx]['H'])} → {int(df_pos.iloc[j]['H'])}",
                'V': f"{int(df.iloc[idx]['V'])} → {int(df_pos.iloc[j]['V'])}",
                'E': f"{int(df.iloc[idx]['E'])} → {int(df_pos.iloc[j]['E'])}"
            })
        st.dataframe(pd.DataFrame(top10), use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Download
        st.subheader("Download")
        
        output_df = model.create_output_dataframe(r)
        
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            output_df.to_excel(writer, sheet_name='Optimized', index=False)
            comp.to_excel(writer, sheet_name='Results', index=False)
        buffer.seek(0)
        
        st.download_button(
            "Download Excel",
            data=buffer,
            file_name=f"SPOI_optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

# ============================================================
# TAB 3
# ============================================================
with tab3:
    st.header("Grafici")
    
    if 'optimized' not in st.session_state:
        st.warning(" Prvo uploadaj fajl i pokreni optimizaciju!")
    else:
        r = st.session_state['results']
        df = r['df']
        df_pos = r['df_positions']
        n = r['n_items']
        opt = r['opt_assign']
        
        C_I, C_O, C_A = '#E74C3C', '#27AE60', '#3498DB'
        izlaz = df['izlaz'].values
        sidx = np.argsort(izlaz)[::-1]
        xi = np.arange(n)
        
        # 1. Summary
        st.subheader(" Summary")
        fig, ax = plt.subplots(2, 2, figsize=(12, 8))
        for a, y, t, c in [(ax[0,0], [r['init_utils'].sum(), r['opt_utils'].sum()], 'Total Utility', C_O),
                          (ax[0,1], [r['init_sim'], r['opt_sim']], 'Simulation Cost', C_I),
                          (ax[1,0], [r['init_wH'], r['opt_wH']], 'Weighted H', C_A),
                          (ax[1,1], [r['init_wV'], r['opt_wV']], 'Weighted V', '#9B59B6')]:
            a.plot(['Početno', 'Optim.'], y, 'o-', lw=3, ms=12, c=c)
            a.fill_between(['Početno', 'Optim.'], 0, y, alpha=0.3, color=c)
            a.set_title(t, fontweight='bold')
            a.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        
        st.markdown("---")
        # Pick-weighted distribucija po V i H pozicijama
        st.subheader("Pick-weighted distribucija po V i H pozicijama")

        df = r['df']
        df_pos = r['df_positions']
        opt = r['opt_assign']
        izlaz = df['izlaz'].values
        total_izlaz = izlaz.sum()
        init_V = df['V'].values
        init_H = df['H'].values
        opt_V = np.array([df_pos.iloc[opt[i]]['V'] for i in range(len(df))])
        opt_H = np.array([df_pos.iloc[opt[i]]['H'] for i in range(len(df))])
        V_levels = sorted(df['V'].unique())
        H_levels = sorted(df['H'].unique())
        init_V_share = [izlaz[init_V == v].sum() / total_izlaz for v in V_levels]
        opt_V_share  = [izlaz[opt_V  == v].sum() / total_izlaz for v in V_levels]
        init_H_share = [izlaz[init_H == h].sum() / total_izlaz for h in H_levels]
        opt_H_share  = [izlaz[opt_H  == h].sum() / total_izlaz for h in H_levels]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
        ax[0].plot(V_levels, init_V_share, marker='o', label='Početno')
        ax[0].plot(V_levels, opt_V_share, marker='o', label='Optimizirano')
        ax[0].set_xlabel("Vertikalni nivo (V)")
        ax[0].set_ylabel("Udio pickova")
        ax[0].set_title("Distribucija pickova po V nivoima")
        ax[0].grid(alpha=0.3)
        ax[0].legend()
        ax[1].plot(H_levels, init_H_share, marker='o', label='Početno')
        ax[1].plot(H_levels, opt_H_share, marker='o', label='Optimizirano')
        ax[1].set_xlabel("Horizontalna pozicija (H)")
        ax[1].set_title("Distribucija pickova po H zonama")
        ax[1].grid(alpha=0.3)
        ax[1].legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        
        
        # 6. Cumulative
        st.subheader(" Kumulativni Utility")
        fig, ax = plt.subplots(figsize=(14, 5))
        cum_i = np.cumsum(r['init_utils'][sidx])
        cum_o = np.cumsum(r['opt_utils'][sidx])
        ax.plot(xi, cum_i, c=C_I, label=f'Početno ({cum_i[-1]:.1f})')
        ax.plot(xi, cum_o, c=C_O, label=f'Optim. ({cum_o[-1]:.1f})')
        ax.fill_between(xi, cum_i, cum_o, alpha=0.3, color=C_O)
        ax.set_xlabel('Artikli'); ax.set_ylabel('Kumulativni Utility')
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


        st.markdown("---")


        st.subheader(" Poboljšanje Utility po Artiklima (%)")
        fig, ax = plt.subplots(figsize=(14, 5))
        improvement_pct = (r['opt_utils'][sidx] - r['init_utils'][sidx]) / np.maximum(r['init_utils'][sidx], 0.001) * 100
        ax.plot(xi, improvement_pct, c=C_O, lw=1.5)
        ax.fill_between(xi, 0, improvement_pct, where=improvement_pct > 0, alpha=0.4, color=C_O, label='Poboljšanje')
        ax.fill_between(xi, 0, improvement_pct, where=improvement_pct < 0, alpha=0.4, color=C_I, label='Pogoršanje')
        ax.axhline(0, c='black', lw=1)
        ax.axhline(np.mean(improvement_pct), c=C_A, ls='--', lw=2, label=f'Prosjek: {np.mean(improvement_pct):.1f}%')
        ax.set_xlabel('Artikli (po potražnji)')
        ax.set_ylabel('Poboljšanje (%)')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")
        
        st.subheader("CDF Simulation Cost (po picku)")
        np.random.seed(42)
        df=r['df']
        izlaz=df['izlaz'].values
        total_izlaz=izlaz.sum()
        probs=izlaz/total_izlaz if total_izlaz>0 else np.ones(len(df))/len(df)
        init_costs=r['init_costs']
        opt_costs=r['opt_costs']
        picked=np.random.choice(len(df),size=r['params']['N_PICKS'],p=probs)
        init_pick_costs=np.sort(init_costs[picked])
        opt_pick_costs=np.sort(opt_costs[picked])
        cdf=np.linspace(0,1,len(init_pick_costs))
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot(init_pick_costs,cdf,label='Početno')
        ax.plot(opt_pick_costs,cdf,label='Optimizirano')
        ax.set_xlabel("Cost po picku")
        ax.set_ylabel("CDF")
        ax.set_title("CDF distribucija simulation cost-a")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

# FOOTER

st.markdown("---")
st.markdown("<div style='text-align:center;color:#6B7280'> SPOI Optimizacija skladišnih pozicija</div>", unsafe_allow_html=True)
