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
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
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
    st.markdown('<h1 class="main-header">🎲 AI Mock Data Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Generate realistic mock data using Groq\'s fast AI models</p>', unsafe_allow_html=True)
    
    # Get API configuration
    api_key, api_url = get_groq_config()
    
    if not api_key:
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Generation Settings")
        
        # Model selection
        model = st.selectbox(
            "Choose Groq Model",
            options=[
                "llama3-8b-8192",
                "llama3-70b-8192"
            ],
            index=0,
            help="llama3-8b-8192 is fastest and recommended for most use cases"
        )
        
        # Number of rows
        num_rows = st.number_input(
            "Number of rows to generate",
            min_value=1,
            max_value=100,
            value=25,
            step=1,
            help="Maximum 100 rows for optimal performance"
        )
        
        # File name
        file_name = st.text_input(
            "CSV filename",
            value="ai_mock_data.csv",
            help="Name for the downloaded CSV file"
        )
        
        st.success("🚀 Groq: Super fast inference with generous free tier!")
        
        st.divider()
        
        st.header("📊 Model Info")
        model_info = {
            "llama3-8b-8192": "Fast, efficient for most tasks",
            "llama3-70b-8192": "More powerful, slower"
        }
        st.info(f"**{model}**: {model_info.get(model, 'High-quality model')}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Data Requirements")
        
        # User prompt
        user_prompt = st.text_area(
            "Describe your data requirements",
            placeholder="e.g., 'Generate customer data for an e-commerce platform. Include demographics, purchase history, and preferences. Make sure ages are realistic for online shoppers.'",
            height=120,
            help="Be specific about the type of data, relationships between fields, constraints, etc."
        )
        
        # Column input
        st.subheader("🏗️ Define Columns")
        columns_input = st.text_area(
            "Enter column names (one per line or comma-separated)",
            placeholder="customer_id\nfirst_name\nlast_name\nemail\nage\ncity\ncountry\ntotal_purchases\npreferred_category\nis_premium_member",
            height=150,
            help="List the exact column names you want in your dataset"
        )
        
        # Parse columns
        if columns_input:
            if '\n' in columns_input:
                columns = [col.strip() for col in columns_input.split('\n') if col.strip()]
            else:
                columns = [col.strip() for col in columns_input.split(',') if col.strip()]
        else:
            columns = []
        
        # Show column preview
        if columns:
            st.write("**Columns to generate:**", ", ".join(columns))
        
        # Generate button
        can_generate = len(columns) > 0
        
        if st.button("🚀 Generate Mock Data", type="primary", disabled=not can_generate):
            if not columns:
                st.error("Please enter at least one column name.")
            else:
                with st.spinner(f"Generating mock data using {model}..."):
                    start_time = time.time()
                    df = generate_mock_data_with_llm(
                        columns, num_rows, user_prompt, model, api_key
                    )
                    generation_time = time.time() - start_time
                
                if not df.empty:
                    st.success(f"Generated {len(df)} rows of mock data in {generation_time:.2f} seconds!")
                    
                    # Store in session state
                    st.session_state['generated_df'] = df
                    st.session_state['generation_time'] = generation_time
    
    with col2:
        st.subheader("💡 Tips & Examples")
        
        with st.expander("🎯 Effective Prompts"):
            st.markdown("""
            **Be Specific:**
            - "Generate employee data for a tech company with realistic salaries based on experience level"
            - "Create customer records with purchase patterns that make business sense"
            
            **Include Constraints:**
            - "Ages should be 18-65 for workplace data"
            - "Ensure email domains match company names"
            - "Make sure dates are chronologically logical"
            
            **Specify Relationships:**
            - "Senior employees should have higher salaries"
            - "Premium customers should have higher purchase amounts"
            """)
        
        with st.expander("🏗️ Column Examples"):
            st.markdown("""
            **Customer Data:**
            ```
            customer_id, first_name, last_name, email,
            phone, age, city, country, signup_date,
            total_spent, order_count, is_vip
            ```
            
            **Employee Data:**
            ```
            employee_id, full_name, department, 
            job_title, hire_date, salary, 
            manager_id, performance_rating
            ```
            
            **Product Data:**
            ```
            product_id, name, category, price,
            cost, supplier, stock_quantity,
            rating, launch_date, is_featured
            ```
            """)
        
        with st.expander("⚠️ Important Notes"):
            st.markdown("""
            - **Fast Generation**: Groq provides super-fast inference
            - **Data Quality**: Review generated data for accuracy
            - **Privacy**: Don't include real personal data in prompts
            - **Limits**: Start with small batches to test
            - **Relationships**: AI understands data relationships
            """)
        
        with st.expander("🎯 Best Practices"):
            st.markdown("""
            **For Better Results:**
            - Use descriptive column names
            - Specify data formats you need
            - Mention business context
            - Include validation rules
            
            **Example Good Prompt:**
            *"Generate realistic e-commerce customer data. Ages 18-70, valid email formats, purchase amounts $10-$5000, signup dates within last 2 years. Make premium customers have higher purchase amounts."*
            """)
    
    # Display generated data if available
    if 'generated_df' in st.session_state:
        st.divider()
        st.subheader("📊 Generated Data")
        
        df = st.session_state['generated_df']
        generation_time = st.session_state.get('generation_time', 0)
        
        # Display preview
        st.dataframe(df, use_container_width=True)
        
        # Download buttons
        col1, col2 = st.columns(2)
        
        with col1:
            # CSV download
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                help="Download the generated data as a CSV file",
                use_container_width=True
            )
        
        with col2:
            # JSON download
            json_data = df.to_json(orient='records', indent=2)
            json_filename = file_name.replace('.csv', '.json')
            
            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                help="Download the generated data as a JSON file",
                use_container_width=True
            )
        
        # Data statistics
        with st.expander("📈 Data Statistics"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", len(df))
            with col2:
                st.metric("Columns", len(df.columns))
            with col3:
                st.metric("Generation Time", f"{generation_time:.2f}s")
            with col4:
                st.metric("Size", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
            
            # Show data types
            st.write("**Column Data Types:**")
            for col in df.columns:
                st.write(f"- **{col}**: {df[col].dtype}")

    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem; padding: 1rem; color: #888; font-size: 0.9rem;">
        <p>Made with ❤️ using Groq's lightning-fast AI models</p>
        <p>No sign-up required • Privacy-focused • Instant generation</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()