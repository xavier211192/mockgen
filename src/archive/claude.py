import streamlit as st
import pandas as pd
import json
import io
import requests
from typing import List, Dict, Any
import time

# LLM API configurations
LLM_PROVIDERS = {
    "OpenAI": {
        "name": "OpenAI",
        "url": "https://api.openai.com/v1/chat/completions",
        "models": ["gpt-4", "gpt-3.5-turbo", "gpt-4-turbo"],
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "free": False,
        "format": "openai"
    },
    "Anthropic": {
        "name": "Anthropic",
        "url": "https://api.anthropic.com/v1/messages",
        "models": ["claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
        "headers": lambda api_key: {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        "free": False,
        "format": "anthropic"
    },
    "Hugging Face": {
        "name": "Hugging Face",
        "url": "https://api-inference.huggingface.co/models",
        "models": [
            "meta-llama/Meta-Llama-3-8B-Instruct",
            "meta-llama/Meta-Llama-3-70B-Instruct", 
            "mistralai/Mistral-7B-Instruct-v0.3",
            "microsoft/DialoGPT-medium",
            "google/gemma-7b-it"
        ],
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "free": True,
        "format": "huggingface"
    },
    "Groq": {
        "name": "Groq",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "models": [
            "llama3-8b-8192",
            "llama3-70b-8192",
            "mixtral-8x7b-32768",
            "gemma-7b-it"
        ],
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "free": True,
        "format": "openai"
    },
    "Together AI": {
        "name": "Together AI", 
        "url": "https://api.together.xyz/v1/chat/completions",
        "models": [
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Llama-3-70b-chat-hf",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "togethercomputer/RedPajama-INCITE-Chat-3B-v1"
        ],
        "headers": lambda api_key: {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        "free": True,
        "format": "openai"
    }
}

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

def call_openai_format_api(provider: str, api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call OpenAI-format APIs (OpenAI, Groq, Together AI)"""
    headers = LLM_PROVIDERS[provider]["headers"](api_key)
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a data generation assistant. Generate realistic mock data in JSON format as requested."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }
    
    response = requests.post(LLM_PROVIDERS[provider]["url"], headers=headers, json=payload)
    return response.json()

def call_anthropic_api(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call Anthropic API"""
    headers = LLM_PROVIDERS["Anthropic"]["headers"](api_key)
    
    payload = {
        "model": model,
        "max_tokens": 4000,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    response = requests.post(LLM_PROVIDERS["Anthropic"]["url"], headers=headers, json=payload)
    return response.json()

def call_huggingface_api(api_key: str, model: str, prompt: str) -> Dict[str, Any]:
    """Call Hugging Face Inference API"""
    model_url = f"{LLM_PROVIDERS['Hugging Face']['url']}/{model}"
    headers = LLM_PROVIDERS["Hugging Face"]["headers"](api_key)
    
    payload = {
        "inputs": f"Generate realistic mock data in JSON format. {prompt}",
        "parameters": {
            "max_new_tokens": 2000,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    response = requests.post(model_url, headers=headers, json=payload)
    return response.json()

def generate_mock_data_with_llm(columns: List[str], num_rows: int, user_prompt: str, 
                               provider: str, model: str, api_key: str) -> pd.DataFrame:
    """Generate mock data using LLM"""
    
    # Create the prompt
    prompt = create_data_generation_prompt(columns, num_rows, user_prompt)
    
    try:
        # Call the appropriate API based on format
        provider_format = LLM_PROVIDERS[provider]["format"]
        
        if provider_format == "openai":
            response = call_openai_format_api(provider, api_key, model, prompt)
            if "choices" in response and len(response["choices"]) > 0:
                content = response["choices"][0]["message"]["content"]
            else:
                raise Exception(f"API Error: {response.get('error', 'Unknown error')}")
                
        elif provider_format == "anthropic":
            response = call_anthropic_api(api_key, model, prompt)
            if "content" in response and len(response["content"]) > 0:
                content = response["content"][0]["text"]
            else:
                raise Exception(f"API Error: {response.get('error', 'Unknown error')}")
                
        elif provider_format == "huggingface":
            response = call_huggingface_api(api_key, model, prompt)
            if isinstance(response, list) and len(response) > 0:
                content = response[0].get("generated_text", "")
            elif isinstance(response, dict) and "generated_text" in response:
                content = response["generated_text"]
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
    st.set_page_config(page_title="LLM Mock Data Generator", page_icon="🤖", layout="wide")
    
    st.title("🤖 LLM-Powered Mock Data Generator")
    st.markdown("Generate realistic mock data using Large Language Models (OpenAI GPT or Anthropic Claude)")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 LLM Configuration")
        
        # Provider selection with free indicators
        provider_options = []
        for key, config in LLM_PROVIDERS.items():
            label = f"{key} {'🆓' if config['free'] else '💰'}"
            provider_options.append((label, key))
        
        provider_display = st.selectbox(
            "Choose LLM Provider",
            options=[option[0] for option in provider_options],
            help="🆓 = Free tier available, 💰 = Paid service"
        )
        
        # Get actual provider key
        provider = next(option[1] for option in provider_options if option[0] == provider_display)
        
        # Model selection
        model = st.selectbox(
            "Choose Model",
            options=LLM_PROVIDERS[provider]["models"],
            help="Select the specific model to use"
        )
        
        # API Key input
        api_key = st.text_input(
            f"{provider} API Key",
            type="password",
            help=f"Enter your {provider} API key"
        )
        
        st.divider()
        
        st.header("⚙️ Generation Settings")
        
        # Number of rows
        num_rows = st.number_input(
            "Number of rows to generate",
            min_value=1,
            max_value=100,  # Limited for API costs
            value=10,
            step=1,
            help="Note: Higher numbers may increase API costs"
        )
        
        # File name
        file_name = st.text_input(
            "CSV filename (optional)",
            value="llm_mock_data.csv",
            help="Name for the downloaded CSV file"
        )
        
        # Cost estimation
        is_free = LLM_PROVIDERS[provider]["free"]
        if is_free:
            if provider == "Groq":
                st.success("🚀 Groq: Super fast inference with generous free tier!")
            elif provider == "Hugging Face":
                st.success("🤗 Hugging Face: Free inference with rate limits")
            elif provider == "Together AI":
                st.success("🤝 Together AI: $25 free credits for new users")
        else:
            if provider == "OpenAI":
                estimated_cost = num_rows * 0.002  # Rough estimate
                st.info(f"Estimated cost: ~${estimated_cost:.3f}")
            elif provider == "Anthropic":
                estimated_cost = num_rows * 0.003  # Rough estimate
                st.info(f"Estimated cost: ~${estimated_cost:.3f}")
    
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
        can_generate = len(columns) > 0 and api_key.strip() != ""
        
        if st.button("🚀 Generate Mock Data", type="primary", disabled=not can_generate):
            if not api_key:
                st.error("Please enter your API key in the sidebar.")
            elif not columns:
                st.error("Please enter at least one column name.")
            else:
                with st.spinner(f"Generating mock data using {provider} {model}..."):
                    start_time = time.time()
                    df = generate_mock_data_with_llm(
                        columns, num_rows, user_prompt, provider, model, api_key
                    )
                    generation_time = time.time() - start_time
                
                if not df.empty:
                    st.success(f"Generated {len(df)} rows of mock data in {generation_time:.2f} seconds!")
                    
                    # Display preview
                    st.subheader("📊 Data Preview")
                    st.dataframe(df, use_container_width=True)
                    
                    # Download button
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=file_name,
                        mime="text/csv",
                        help="Download the generated data as a CSV file"
                    )
                    
                    # Data statistics
                    with st.expander("📈 Data Statistics"):
                        st.write(f"**Rows:** {len(df)}")
                        st.write(f"**Columns:** {len(df.columns)}")
                        st.write(f"**Memory usage:** {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
                        st.write(f"**Generation time:** {generation_time:.2f} seconds")
                        
                        # Show data types
                        st.write("**Column Data Types:**")
                        for col in df.columns:
                            st.write(f"- {col}: {df[col].dtype}")
    
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
            - **API Costs**: Each generation uses your API quota
            - **Rate Limits**: Respect provider rate limits
            - **Data Quality**: Review generated data for accuracy
            - **Privacy**: Don't include real personal data in prompts
            - **Limits**: Start with small batches to test
            """)
        
        with st.expander("🔑 Getting API Keys"):
            st.markdown("""
            **🆓 FREE OPTIONS:**
            
            **Groq (Recommended):**
            1. Visit [console.groq.com](https://console.groq.com)
            2. Sign up for free account
            3. Generate API key (generous free tier)
            
            **Hugging Face:**
            1. Visit [huggingface.co](https://huggingface.co)
            2. Create account and go to Settings → Access Tokens
            3. Create new token (free tier with limits)
            
            **Together AI:**
            1. Visit [api.together.xyz](https://api.together.xyz)
            2. Sign up for $25 free credits
            3. Get API key from dashboard
            
            **💰 PAID OPTIONS:**
            
            **OpenAI:**
            1. Visit [platform.openai.com](https://platform.openai.com)
            2. Sign up/login and go to API keys
            3. Create a new secret key
            
            **Anthropic:**
            1. Visit [console.anthropic.com](https://console.anthropic.com)
            2. Sign up/login 
            3. Generate an API key
            """)

if __name__ == "__main__":
    main()