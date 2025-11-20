import streamlit as st
import pandas as pd
import plotly.express as px
from main import documents, hybrid_retrieve, get_answer

# Page config
st.set_page_config(
    page_title="Medical Claims Analytics",
    page_icon="🏥",
    layout="wide"
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Claim Search", "Chat Analysis"])

# Dashboard Page
if page == "Dashboard":
    st.title("🏥 Medical Claims Analytics Dashboard")
    
    # Load your CSV data for analytics
    try:
        df = pd.read_csv("data/kaiser_medical_claims.csv")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Claims", len(df))
        with col2:
            st.metric("Total Billed", f"${df['Amount Billed ($)'].sum():,}")
        with col3:
            approved = len(df[df['Claim Status'] == 'Approved'])
            st.metric("Approved Claims", approved)
        with col4:
            st.metric("Approval Rate", f"{(approved/len(df)*100):.1f}%")
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            # Claims by status
            status_counts = df['Claim Status'].value_counts()
            fig1 = px.pie(values=status_counts.values, names=status_counts.index, 
                         title="Claims by Status")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Claims by provider
            provider_counts = df['Provider'].value_counts().head(5)
            fig2 = px.bar(x=provider_counts.values, y=provider_counts.index,
                         title="Top 5 Providers by Claim Volume", orientation='h')
            st.plotly_chart(fig2, use_container_width=True)
        
        # Data table
        st.subheader("Claims Data")
        st.dataframe(df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Claim Search Page
elif page == "Claim Search":
    st.title("🔍 Claim Search & Analysis")
    
    search_query = st.text_input("Search claims by patient, diagnosis, or procedure:")
    
    if search_query:
        # Use your RAG system to find relevant claims
        candidates = hybrid_retrieve(search_query, top_k=10)
        
        st.subheader(f"Found {len(candidates)} relevant claims:")
        for i, claim in enumerate(candidates, 1):
            with st.expander(f"Claim {i}"):
                st.write(claim)

# Chat Analysis Page
elif page == "Chat Analysis":
    st.title("💬 Chat Analysis")
    
    st.info("This shows how the RAG system processes queries")
    
    query = st.text_input("Enter a query to see RAG processing:")
    if query:
        # Show the retrieval process
        from main import hybrid_retrieve, rerank_with_crossencoder
        
        candidates = hybrid_retrieve(query, top_k=6)
        reranked = rerank_with_crossencoder(query, candidates, top_k=3)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Initial Retrieval")
            for i, doc in enumerate(candidates, 1):
                st.write(f"**{i}.** {doc[:200]}...")
        
        with col2:
            st.subheader("After Re-ranking")
            for i, doc in enumerate(reranked, 1):
                st.write(f"**{i}.** {doc[:200]}...")
        
        # Show final answer
        st.subheader("Final Answer")
        answer = get_answer(query)
        st.write(answer)