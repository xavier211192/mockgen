import streamlit as st
import json
import pandas as pd
import tempfile
import subprocess
import requests
import re

st.set_page_config(page_title="Mock Data Generator (Groq + Secrets)", layout="centered")
st.title("🧪 Mock Data Generator (Groq + Faker)")
st.markdown("Generate test CSV data from a JSON schema using Groq's Mixtral model and Faker.")

# Load Groq API key from Streamlit secrets
try:
    groq_key = st.secrets["groq"]["api_key"]
except Exception:
    st.error("Groq API key not found in Streamlit secrets! Please add it to `.streamlit/secrets.toml` locally or via Streamlit Cloud dashboard.")
    st.stop()

# Example schema and instructions
EXAMPLE_SCHEMA = '''{
  "fields": [
    { "name": "user_id", "type": "uuid", "unique": true },
    { "name": "full_name", "type": "string" },
    { "name": "email", "type": "string", "unique": true },
    { "name": "signup_date", "type": "datetime", "start": "2023-01-01", "step": "1d" },
    { "name": "country", "type": "string" },
    { "name": "is_active", "type": "boolean" }
  ],
  "rows": 25
}'''

EXAMPLE_PROMPT = '''- Ensure each `signup_date` is a unique daily timestamp starting from 2023-01-01.
- Each email should look realistic and match the full name.
- Use a mix of countries like "USA", "India", "Germany", "Brazil", "Japan".
- Make about 70% of `is_active` values True.'''

if st.button("Use Example"):
    st.session_state.schema = EXAMPLE_SCHEMA
    st.session_state.prompt = EXAMPLE_PROMPT

# schema_input = st.text_area("Enter JSON Schema", value=st.session_state.get("schema", ""), height=250)
# instructions = st.text_area("Additional Instructions (Prompt)", value=st.session_state.get("prompt", ""), height=200)

# Inputs
schema_input = st.text_area("📄 JSON Schema",value=st.session_state.get("schema", ""), height=300, placeholder="""
{
  "fields": [
    { "name": "name", "type": "string" },
    { "name": "email", "type": "string", "unique": true },
    { "name": "signup_date", "type": "datetime", "start": "2023-01-01", "step": "1d" }
  ],
  "rows": 50
}
""")

instructions = st.text_area("📝 Additional Instructions (Optional)", value=st.session_state.get("prompt", ""),placeholder="E.g., 'Ensure signup_date is unique and incremented daily from 2023-01-01.'")

if st.button("🚀 Generate Data"):
    if not schema_input:
        st.warning("Please provide the JSON schema.")
    else:
        try:
            schema = json.loads(schema_input)
            row_count = schema.get("rows", 10)
            fields = schema.get("fields", [])

            # Build the prompt
            field_lines = "\n".join([f"- {f['name']} ({f['type']})" for f in fields])
            prompt = f"""
Generate a Python script using the Faker and pandas libraries.

Requirements:
- Generate {row_count} rows of data
- Fields:
{field_lines}

{instructions or ''}

The script should save the DataFrame to 'output.csv' using df.to_csv('output.csv', index=False).
Only return the code. No explanation.
"""

            # Call Groq API
            st.info("Calling Groq (Mixtral) to generate code...")
            headers = {
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4
            }

            response = requests.post("https://api.groq.com/openai/v1/chat/completions",
                                     headers=headers, json=payload)

            result = response.json()
            st.write("Groq response received.")
            st.write(result)  # prints the raw API response
            content = result["choices"][0]["message"]["content"]

            # Extract Python code from markdown formatting
            match = re.search(r"```(?:python)?\n(.*?)```", content, re.DOTALL)
            code = match.group(1) if match else content

            # Write and run generated script
            with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as tmp:
                tmp.write(code)
                script_path = tmp.name

            subprocess.run(["python", script_path], check=True, timeout=20)

            # Load and show the generated CSV
            df = pd.read_csv("output.csv")
            st.success("✅ Data generated successfully!")
            st.dataframe(df)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "mock_data.csv", "text/csv")

        except Exception as e:
            st.error(f"❌ Error: {e}")