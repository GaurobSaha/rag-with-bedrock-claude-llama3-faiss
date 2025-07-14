import boto3
import json
from botocore.exceptions import ClientError

# Creating a Bedrock Runtime client in my AWS region
client = boto3.client("bedrock-runtime", region_name="us-east-1")

# LLaMA 3 8B Instruct model ID on Bedrock
model_id = "meta.llama3-8b-instruct-v1:0"

#  user prompt
prompt = "Write a poetry on a programmer's life."

# Formatting the prompt in LLaMA 3's instruction format
formatted_prompt = f"""
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""

# Define the payload in native format
payload = {
    "prompt": formatted_prompt,
    "max_gen_len": 256,
    "temperature": 0.5
}

# Convert the payload to JSON string
body = json.dumps(payload)

try:
    # Send the request to the model
    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    
    # Parse the response
    model_response = json.loads(response['body'].read())
    print("=== Model Response ===")
    print(model_response['generation'])

except ClientError as e:
    print(f"AWS ClientError: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
