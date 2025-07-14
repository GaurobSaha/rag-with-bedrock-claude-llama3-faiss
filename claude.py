import boto3
import json
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-v2"
prompt = "Write a poetry on a programmer's life."

payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 128,
    "temperature": 0.5,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]
}

body = json.dumps(payload)

try:
    response = client.invoke_model(
        modelId=model_id,
        body=body,
        contentType="application/json",
        accept="application/json"
    )
    result = json.loads(response["body"].read())
    generated_text = result["content"][0]["text"]

    # Print and save to file
    print("=== Claude Output ===")
    print(generated_text)

    with open("claude_output.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

except ClientError as e:
    print(f"AWS ClientError: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
