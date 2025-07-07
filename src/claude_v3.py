import streamlit as st
import pandas as pd
import json
import io
import requests
from typing import List, Dict, Any
import time
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", message=".*ScriptRunContext.*")

# Page configuration
st.set_page_config(
    page_title="AI Mock Data Generator",
    page_icon="🎲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling with colors
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Global styling */
    .main {
        font-family: 'Inter', sans-serif;
        max-width: 900px;
        margin: 0 auto;
        padding: 0 1rem;
    }
    
    /* Container for content */
    .main .block-container {
        max-width: 900px;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    .main-header h1 {
        font-size: 2.25rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Card styling */
    .card {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.3);
        border: 1px solid #6366f1;
        margin-bottom: 1.5rem;
    }
    
    .card h4 {
        color: white;
        margin: 0 0 1rem 0;
        font-weight: 600;
    }
    
    .card p {
        color: rgba(255, 255, 255, 0.95);
        margin: 0.5rem 0;
    }
    
    .card em {
        color: rgba(255, 255, 255, 0.9);
        font-style: italic;
    }
    
    /* Mobile optimization for cards */
    @media (max-width: 768px) {
        .card {
            background: linear-gradient(135deg, #3730a3 0%, #6b21a8 100%);
            padding: 1.25rem;
            box-shadow: 0 6px 20px rgba(55, 48, 163, 0.4);
            border: 2px solid #4f46e5;
        }
        
        .card h4 {
            font-size: 1.1rem;
            font-weight: 700;
        }
        
        .card p {
            font-size: 0.95rem;
            line-height: 1.5;
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
        border-right: 1px solid #e5e7eb;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.25);
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.35);
        transform: translateY(-1px);
    }
    
    /* Form styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
        transition: border-color 0.2s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        outline: none;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    .stSelectbox select {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }
    
    .stNumberInput input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }
    
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e5e7eb;
        font-family: 'Inter', sans-serif;
        font-size: 0.875rem;
    }
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    .metric-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.75rem;
        color: #6b7280;
        margin: 0.25rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    /* Success/Error styling */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-size: 0.875rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        font-weight: 500;
        font-size: 0.875rem;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #4f46e5;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #4f46e5;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* Mobile optimizations for section headers */
    @media (max-width: 768px) {
        .section-header {
            font-size: 1.35rem;
            font-weight: 700;
            color: #4338ca;
            padding: 0.75rem;
            background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
            border-radius: 8px;
            border-bottom: 3px solid #4338ca;
            box-shadow: 0 2px 8px rgba(67, 56, 202, 0.15);
        }
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin-top: 3rem;
        border: 1px solid #4b5563;
        box-shadow: 0 4px 15px rgba(31, 41, 55, 0.3);
    }
    
    .footer h3 {
        color: white;
        margin: 0 0 1rem 0;
        font-weight: 600;
    }
    
    .footer p {
        color: rgba(255, 255, 255, 0.9);
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Mobile optimization for footer */
    @media (max-width: 768px) {
        .footer {
            background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
            padding: 1.5rem;
            margin-top: 2rem;
            border: 2px solid #374151;
            box-shadow: 0 6px 20px rgba(17, 24, 39, 0.4);
        }
        
        .footer h3 {
            font-size: 1.2rem;
            font-weight: 700;
            margin-bottom: 1rem;
        }
        
        .footer p {
            font-size: 0.9rem;
            line-height: 1.5;
            color: rgba(255, 255, 255, 0.95);
        }
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Code blocks */
    .stCode {
        background: #f3f4f6;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
    
    /* Clean dataframe */
    .dataframe {
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    }
</style>
""", unsafe_allow_html=True)

def get_groq_config():
    """Get Groq API configuration from secrets"""
    try:
        api_key = st.secrets["groq"]["GROQ_API_KEY"]
        return api_key, "https://api.groq.com/openai/v1/chat/completions"
    except KeyError:
        st.error("⚠️ GROQ_API_KEY not found in secrets. Please configure it.")
        return None, None

def create_data_generation_prompt(columns: List[str], num_rows: int, user_prompt: str = "") -> str:
    """Create a prompt for the LLM to generate mock data"""
    
    base_prompt = f"""Generate {num_rows} rows of realistic mock data in JSON format.

COLUMNS: {', '.join(columns)}

USER REQUIREMENTS: {user_prompt if user_prompt else "Generate realistic and diverse data"}

IMPORTANT INSTRUCTIONS:
1. Return ONLY a valid JSON array of objects
2. Each object should have exactly these keys: {columns}
3. Make the data realistic and diverse
4. Use appropriate data types (strings, numbers, booleans, dates as needed)
5. Ensure data consistency (e.g., if there's a country and city, make sure they match)
6. For dates, use YYYY-MM-DD format
7. For names, use realistic full names unless specified otherwise
8. For emails, create realistic email addresses
9. For IDs, use consistent formatting within the dataset

Example format:
[
  {{"column1": "value1", "column2": "value2"}},
  {{"column1": "value3", "column2": "value4"}}
]

Generate the data now:"""
    
    return base_prompt

def call_groq_api(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call Groq API"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a data generation assistant. Generate realistic mock data in JSON format as requested."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    try:
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {str(e)}")
        return {"error": str(e)}
    except Exception as e:
        st.error(f"Unexpected Error: {str(e)}")
        return {"error": str(e)}

def generate_mock_data_with_llm(columns: List[str], num_rows: int, user_prompt: str, model: str, api_key: str) -> pd.DataFrame:
    """Generate mock data using Groq LLM"""
    
    # Create the prompt
    prompt = create_data_generation_prompt(columns, num_rows, user_prompt)
    
    try:
        # Call Groq API
        response = call_groq_api(api_key, model, prompt)
        
        if "error" in response:
            raise Exception(f"API Error: {response['error']}")
        
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
        else:
            raise Exception(f"API Error: {response.get('error', 'Unknown error')}")
        
        # Clean the response to extract JSON
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON array in the response
            import re
            json_match = re.search(r'\[.*\]', content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise Exception("Could not parse JSON from LLM response")
        
        # Convert to DataFrame
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            
            # Ensure all requested columns are present
            for col in columns:
                if col not in df.columns:
                    df[col] = "N/A"
            
            # Reorder columns to match request
            df = df[columns]
            return df
        else:
            raise Exception("No valid data generated")
            
    except Exception as e:
        st.error(f"Error generating data: {str(e)}")
        return pd.DataFrame()

def main():
    # Header with gradient
    st.markdown("""
    <div class="main-header">
        <h1>🎲 AI Mock Data Generator</h1>
        <p>Generate realistic test data using advanced AI models</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem; opacity: 0.9;">
            💻 For best experience, open on desktop
        </p>
        <div style="margin-top: 1.5rem; padding: 1rem; background: #f3f4f6; border-radius: 8px;">
        <p style="margin: 0; font-weight: 500; color: #374151;">
            <a href="https://forms.gle/pfGWBnDffyoW6tRN9" target="_blank" style="color: #667eea; text-decoration: none;">
            🔬 Testing prototype - Need your input for official launch! Share your feedback in our quick 2-minute survey
            </a>
        </p>
    </div>        
    </div>
    """, unsafe_allow_html=True)
    
    # Get API configuration
    api_key, api_url = get_groq_config()
    
    if not api_key:
        st.stop()
    
    # Sidebar - Note: If closed, refresh the page to reopen
    with st.sidebar:
        st.markdown("### ⚙️ Generation Settings")
        st.info("💡 If this sidebar closes, refresh the page to reopen it")
        
        # Model selection
        model = st.selectbox(
            "🤖 AI Model",
            options=[
                "llama3-8b-8192",
                "llama3-70b-8192"
            ],
            index=0,
            help="Choose the AI model for generation"
        )
        
        # Number of rows
        num_rows = st.number_input(
            "📊 Number of Rows",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
            help="Maximum 100 rows for optimal performance"
        )
        
        # File name
        file_name = st.text_input(
            "📁 Filename",
            value="mock_data",
            help="Name for downloaded files"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📈 Model Information")
        model_info = {
            "llama3-8b-8192": {
                "name": "Llama 3 8B", 
                "speed": "⚡ Very Fast",
                "quality": "🎯 High Quality",
                "desc": "Recommended for most use cases"
            },
            "llama3-70b-8192": {
                "name": "Llama 3 70B",
                "speed": "🐌 Slower", 
                "quality": "🏆 Highest Quality",
                "desc": "Best for complex data relationships"
            }
        }
        
        info = model_info.get(model, {})
        if info:
            st.markdown(f"""
            <div class="card">
                <h4>{info['name']}</h4>
                <p><strong>Speed:</strong> {info['speed']}</p>
                <p><strong>Quality:</strong> {info['quality']}</p>
                <p><em>{info['desc']}</em></p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.success("✅ Service Ready")
        st.info("🚀 Fast AI-powered generation")
    
    # Main content
    col1, col2 = st.columns([2, 1], gap="large")
    
    with col1:
        # Data requirements section
        st.markdown('<div class="section-header">📝 Data Requirements</div>', unsafe_allow_html=True)
        
        # Description input
        user_prompt = st.text_area(
            "💭 Describe your data needs",
            placeholder="e.g., 'Generate customer data for an e-commerce platform. Include demographics, purchase history, and preferences. Make sure ages are realistic for online shoppers and premium customers have higher spending.'",
            height=120,
            help="Be specific about relationships, constraints, and business logic"
        )
        
        # Column input
        st.markdown('<div class="section-header">🏗️ Column Definition</div>', unsafe_allow_html=True)
        columns_input = st.text_area(
            "📋 Column names (one per line or comma-separated)",
            placeholder="customer_id\nfirst_name\nlast_name\nemail\nage\ncity\ncountry\ntotal_purchases\npreferred_category\nis_premium_member",
            height=150,
            help="List the exact column names for your dataset"
        )
        
        # Parse columns
        if columns_input:
            if '\n' in columns_input:
                columns = [col.strip() for col in columns_input.split('\n') if col.strip()]
            else:
                columns = [col.strip() for col in columns_input.split(',') if col.strip()]
        else:
            columns = []
        
        # Column preview
        if columns:
            st.markdown("**📊 Columns to generate:**")
            cols_display = ", ".join([f"`{col}`" for col in columns])
            st.markdown(cols_display)
        
        # Generate button
        can_generate = len(columns) > 0
        
        st.markdown("---")
        
        if st.button("🚀 Generate Data", type="primary", disabled=not can_generate, use_container_width=True):
            if not columns:
                st.error("Please enter at least one column name.")
            else:
                with st.spinner("🤖 AI is generating your data..."):
                    start_time = time.time()
                    df = generate_mock_data_with_llm(
                        columns, num_rows, user_prompt, model, api_key
                    )
                    generation_time = time.time() - start_time
                
                if not df.empty:
                    st.success(f"✅ Generated {len(df)} rows in {generation_time:.2f} seconds!")
                    
                    # Store in session state
                    st.session_state['generated_df'] = df
                    st.session_state['generation_time'] = generation_time
                    st.session_state['file_name'] = file_name
                    st.session_state['model_used'] = model
    
    with col2:
        # Tips and examples
        st.markdown('<div class="section-header">💡 Tips & Examples</div>', unsafe_allow_html=True)
        
        with st.expander("🎯 Writing Effective Prompts"):
            st.markdown("""
            **✅ Be Specific:**
            - "Generate tech company employees with realistic salaries based on experience"
            - "Create customers with purchase patterns that make business sense"
            
            **✅ Include Constraints:**
            - "Ages should be 18-65 for workplace data"
            - "Ensure email domains match company names"
            - "Make dates chronologically logical"
            
            **✅ Define Relationships:**
            - "Senior employees should have higher salaries"
            - "Premium customers should have higher spending"
            """)
        
        with st.expander("📋 Column Name Examples"):
            st.markdown("""
            **👥 Customer Data:**
            ```
            customer_id, first_name, last_name, email,
            phone, age, city, country, signup_date,
            total_spent, order_count, is_premium
            ```
            
            **💼 Employee Data:**
            ```
            employee_id, full_name, department, 
            job_title, hire_date, salary, 
            manager_id, performance_rating
            ```
            
            **📦 Product Data:**
            ```
            product_id, name, category, price,
            cost, supplier, stock_quantity,
            rating, launch_date, is_featured
            ```
            """)
        
        with st.expander("🚀 Best Practices"):
            st.markdown("""
            **🎯 For Better Results:**
            - Use descriptive column names
            - Specify data formats you need
            - Mention business context
            - Include validation rules
            
            **💡 Example:**
            *"Generate realistic e-commerce customer data. Ages 18-70, valid email formats, purchase amounts $10-$5000, signup dates within last 2 years."*
            """)
        
        with st.expander("⚡ About This Tool"):
            st.markdown("""
            **🔥 Features:**
            - Lightning-fast AI generation
            - Realistic data relationships
            - Consistent business logic
            - Multiple export formats
            
            **🛡️ Privacy:**
            - No data stored permanently
            - Secure API connections
            - Generate offline-ready data
            """)
    
    # Display generated data
    if 'generated_df' in st.session_state:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 Generated Data</div>', unsafe_allow_html=True)
        
        df = st.session_state['generated_df']
        generation_time = st.session_state.get('generation_time', 0)
        filename = st.session_state.get('file_name', 'mock_data')
        model_used = st.session_state.get('model_used', 'llama3-8b-8192')
        
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df)}</div>
                <div class="metric-label">Rows Generated</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{len(df.columns)}</div>
                <div class="metric-label">Columns</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{generation_time:.2f}s</div>
                <div class="metric-label">Generation Time</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{df.memory_usage(deep=True).sum() / 1024:.1f} KB</div>
                <div class="metric-label">Data Size</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Data preview
        st.markdown("#### 👀 Data Preview")
        st.dataframe(df, use_container_width=True, height=400)
        
        # Download section
        st.markdown("#### ⬇️ Download Your Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv_data,
                file_name=f"{filename}.csv",
                mime="text/csv",
                help="Perfect for Excel, Google Sheets, and data analysis",
                use_container_width=True
            )
        
        with col2:
            # JSON download
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="🔧 Download JSON",
                data=json_data,
                file_name=f"{filename}.json",
                mime="application/json",
                help="Ideal for APIs, web development, and programming",
                use_container_width=True
            )
        
        # Additional details
        with st.expander("📈 Generation Details"):
            st.markdown(f"""
            **Model Used:** {model_used}  
            **Generation Time:** {generation_time:.2f} seconds  
            **Total Rows:** {len(df)}  
            **Total Columns:** {len(df.columns)}  
            **Memory Usage:** {df.memory_usage(deep=True).sum() / 1024:.1f} KB
            """)
            
            st.markdown("**Column Data Types:**")
            for col in df.columns:
                st.markdown(f"- **{col}**: {df[col].dtype}")

    # Footer
    st.markdown("""
    <div class="footer">
        <h3>🎲 Mock Data Generator Prototype Only</h3>
        <p>Powered by Groq for realistic data generation</p>
        <p>Choose between OpenAI and HuggingFace models in the future edition</p>
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()