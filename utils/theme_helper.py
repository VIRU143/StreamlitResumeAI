import streamlit as st

def initialize_theme():
    """Initialize theme in session state"""
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'

def toggle_theme():
    """Toggle between light and dark theme"""
    st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'

def get_theme_css(theme):
    """Get CSS for the selected theme"""
    if theme == 'dark':
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            .main-header {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                font-family: 'Inter', sans-serif;
                animation: fadeInDown 0.8s ease-in-out;
            }
            
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .stApp {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            }
            
            div[data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 700;
                color: #a78bfa;
            }
            
            div[data-testid="stMetricLabel"] {
                color: #cbd5e1;
                font-weight: 600;
            }
            
            .stButton>button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                border: none;
                padding: 0.75rem 2rem;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            div.row-widget.stRadio > div {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 1rem;
                backdrop-filter: blur(10px);
            }
            
            .uploadedFile {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                padding: 1rem;
            }
            
            h1, h2, h3 {
                font-family: 'Inter', sans-serif;
                color: #e2e8f0;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: rgba(255, 255, 255, 0.05);
                padding: 0.5rem;
                border-radius: 12px;
            }
            
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                color: #cbd5e1;
                font-weight: 600;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .stProgress > div > div > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
        </style>
        """
    else:
        return """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            .main-header {
                font-size: 3.5rem;
                font-weight: 800;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                text-align: center;
                margin-bottom: 2rem;
                font-family: 'Inter', sans-serif;
                animation: fadeInDown 0.8s ease-in-out;
            }
            
            @keyframes fadeInDown {
                from {
                    opacity: 0;
                    transform: translateY(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            
            div[data-testid="stMetricValue"] {
                font-size: 2rem;
                font-weight: 700;
                color: #6366f1;
            }
            
            div[data-testid="stMetricLabel"] {
                color: #475569;
                font-weight: 600;
            }
            
            .stButton>button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white !important;
                border: none;
                padding: 0.75rem 2rem;
                border-radius: 12px;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
            }
            
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
            }
            
            div.row-widget.stRadio > div {
                background: white;
                border-radius: 12px;
                padding: 1rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            .uploadedFile {
                background: white;
                border-radius: 12px;
                padding: 1rem;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            h1, h2, h3 {
                font-family: 'Inter', sans-serif;
                color: #1e293b;
            }
            
            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background: white;
                padding: 0.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            .stTabs [data-baseweb="tab"] {
                border-radius: 8px;
                color: #64748b;
                font-weight: 600;
            }
            
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            
            .stProgress > div > div > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
        </style>
        """

def add_theme_toggle():
    """Add theme toggle button to sidebar"""
    st.sidebar.markdown("### ⚙️ Settings")
    
    theme_emoji = "🌙" if st.session_state.theme == 'light' else "☀️"
    theme_text = "Dark Mode" if st.session_state.theme == 'light' else "Light Mode"
    
    if st.sidebar.button(f"{theme_emoji} {theme_text}", use_container_width=True):
        toggle_theme()
        st.rerun()
    
    st.sidebar.markdown("---")

def apply_theme():
    """Apply theme to page"""
    initialize_theme()
    st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)
    add_theme_toggle()
