import os
import httpx
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def api_get(path: str):
    url = f"{API_BASE_URL}{path}"
    try:
        r = httpx.get(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, data: dict | None = None):
    url = f"{API_BASE_URL}{path}"
    try:
        r = httpx.post(url, json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_delete(path: str):
    url = f"{API_BASE_URL}{path}"
    try:
        r = httpx.delete(url, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error: {e}")
        return None


st.set_page_config(page_title="Pricing Intelligence Agent", layout="wide")
st.title("📊 Competitive Pricing Intelligence Agent")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Products", "Analysis", "Offers", "Settings"])

# ─── Tab 1: Dashboard ──────────────────────────────────────────────
with tab1:
    st.subheader("System Overview")

    health = api_get("/health")
    if health:
        cols = st.columns(3)
        cols[0].metric("Status", health.get("status", "unknown").upper())
        cols[1].metric("Database", health.get("database", "unknown"))
        cols[2].metric("Redis", health.get("redis", "unknown"))

    metrics = api_get("/metrics-summary")
    if metrics:
        st.subheader("Key Metrics")
        cols = st.columns(4)
        cols[0].metric("Tracked Products", metrics.get("tracked_products", 0))
        cols[1].metric("Total Offers", metrics.get("total_offers", 0))
        cols[2].metric("Analyses Run", metrics.get("total_analyses", 0))
        cols[3].metric("Today", metrics.get("analyses_today", 0))

        cols = st.columns(4)
        cols[0].metric("Avg Latency", f"{metrics.get('avg_latency_ms', 0):.0f} ms" if metrics.get("avg_latency_ms") else "N/A")
        cols[1].metric("Match Rate", f"{metrics.get('match_rate', 0):.0%}" if metrics.get("match_rate") else "N/A")
        cols[2].metric("No-Match Rate", f"{metrics.get('no_match_rate', 0):.0%}" if metrics.get("no_match_rate") else "N/A")
        cols[3].metric("Avg Confidence", f"{metrics.get('avg_confidence', 0):.0%}" if metrics.get("avg_confidence") else "N/A")

    dash = api_get("/api/v1/dashboard/summary")
    if dash:
        st.subheader("Recent Analyses")
        recent = dash.get("recent_analyses", [])
        if recent:
            df = pd.DataFrame(recent)
            if not df.empty:
                fig = px.bar(df, x="date", y="score", color="confidence",
                             title="Recent Analysis Scores",
                             labels={"date": "Date", "score": "Best Match Score"})
                st.plotly_chart(fig, use_container_width=True)

        if dash.get("best_price_drops"):
            st.subheader("Best Prices Found")
            df2 = pd.DataFrame(dash["best_price_drops"])
            st.dataframe(df2, use_container_width=True)


# ─── Tab 2: Products ───────────────────────────────────────────────
with tab2:
    st.subheader("Products")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ New Product", use_container_width=True):
            st.session_state.show_add_form = True

    products = api_get("/api/v1/products")
    if products:
        df = pd.DataFrame(products)
        for c in ["created_at", "updated_at"]:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
        st.dataframe(df, use_container_width=True, column_order=["name", "brand", "category", "target_price", "currency", "is_tracked", "created_at"])

        st.subheader("Product Actions")
        sel = st.selectbox("Select a product", options=[p["id"] for p in products],
                           format_func=lambda x: next((p["name"] for p in products if p["id"] == x), x))
        if sel:
            col1, col2, col3 = st.columns(3)
            if col1.button("🔍 Analyze Now"):
                resp = api_post(f"/api/v1/products/{sel}/analyze")
                if resp:
                    st.success(f"Analysis started! Run ID: {resp['run_id']}")
            if col2.button("🗑️ Delete"):
                api_delete(f"/api/v1/products/{sel}")
                st.rerun()

    if st.session_state.get("show_add_form"):
        with st.form("add_product"):
            st.write("Add New Product")
            name = st.text_input("Product Name *")
            desc = st.text_area("Description")
            brand = st.text_input("Brand")
            category = st.text_input("Category")
            target_price = st.number_input("Target Price", min_value=0.0, step=0.01)
            currency = st.selectbox("Currency", ["USD", "EUR", "GBP", "CAD"], index=0)
            if st.form_submit_button("Save") and name:
                api_post("/api/v1/products", {
                    "name": name, "description": desc, "brand": brand,
                    "category": category, "target_price": target_price, "currency": currency,
                })
                st.success("Product created!")
                st.session_state.show_add_form = False
                st.rerun()


# ─── Tab 3: Analysis ───────────────────────────────────────────────
with tab3:
    st.subheader("Analysis History")

    products = api_get("/api/v1/products")
    if products:
        sel = st.selectbox("Select product to view analysis",
                           options=[p["id"] for p in products],
                           format_func=lambda x: next((p["name"] for p in products if p["id"] == x), x),
                           key="analysis_product")
        if sel:
            latest = api_get(f"/api/v1/products/{sel}/analysis/latest")
            if latest:
                st.json(latest.get("final_decision", {}))

                metrics_cols = st.columns(4)
                metrics_cols[0].metric("Status", latest.get("status", "").upper())
                metrics_cols[1].metric("Candidates", latest.get("candidate_count", 0))
                metrics_cols[2].metric("Valid Matches", latest.get("valid_match_count", 0))
                metrics_cols[3].metric("Confidence", f"{latest.get('price_confidence', 0):.0%}" if latest.get("price_confidence") else "N/A")

                if latest.get("total_latency_ms"):
                    st.metric("Total Latency", f"{latest['total_latency_ms']:.0f} ms")

                history = api_get(f"/api/v1/products/{sel}/analysis")
                if history and len(history) > 1:
                    df = pd.DataFrame(history)
                    if "created_at" in df.columns:
                        df["created_at"] = pd.to_datetime(df["created_at"])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=df["created_at"], y=df.get("valid_match_count", 0),
                                                  mode="lines+markers", name="Valid Matches"))
                        fig.update_layout(title="Match History Over Time", xaxis_title="Date", yaxis_title="Matches")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No analysis data for this product yet. Go to Products tab and click 'Analyze Now'.")

            price_history = api_get(f"/api/v1/products/{sel}/price-history?days=30")
            if price_history:
                df = pd.DataFrame(price_history)
                if not df.empty:
                    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
                    fig = px.line(df, x="snapshot_date", y="price", title="Price History (30 days)")
                    st.plotly_chart(fig, use_container_width=True)


# ─── Tab 4: Offers ─────────────────────────────────────────────────
with tab4:
    st.subheader("Competitor Offers")

    products = api_get("/api/v1/products")
    if products:
        sel = st.selectbox("Select product to view offers",
                           options=[p["id"] for p in products],
                           format_func=lambda x: next((p["name"] for p in products if p["id"] == x), x),
                           key="offers_product")
        if sel:
            offers = api_get(f"/api/v1/products/{sel}/offers")
            if offers:
                df = pd.DataFrame(offers)

                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(df[["title", "price", "merchant", "source", "discovered_at"]],
                                 use_container_width=True)

                with col2:
                    if not df.empty and "price" in df.columns:
                        fig = px.bar(df, x="merchant", y="price", color="source",
                                     title="Competitor Prices", text="price")
                        fig.update_traces(texttemplate="$%{text}", textposition="outside")
                        st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No offers found. Run an analysis first.")


# ─── Tab 5: Settings ───────────────────────────────────────────────
with tab5:
    st.subheader("System Configuration")
    st.code(f"API Base URL: {API_BASE_URL}")
    st.code(f"LLM Provider: {os.getenv('LLM_PROVIDER', 'llamacpp (Groq)')}")
    st.code(f"Model: {os.getenv('LLAMA_CPP_MODEL', 'llama-3.1-8b-instant')}")

    if st.button("🔄 Check Health"):
        h = api_get("/health")
        if h:
            st.json(h)

    if st.button("📊 Refresh Metrics"):
        m = api_get("/metrics-summary")
        if m:
            st.json(m)
