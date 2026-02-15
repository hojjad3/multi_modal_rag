import os

from ollama import ChatResponse, chat

image_path = "images/Gemini_Generated_Image_dkbrzgdkbrzgdkbr.png"
file_name = os.path.splitext(os.path.basename(image_path))[0]
file_md = file_name + ".md"
system_prompt = """
Act as a senior data analyst.
Analyze this chart image and extract the raw data values precisely.
Output the result ONLY as a Markdown table.
Do not add any explanation or conversational filler.
"""
response: ChatResponse = chat(
    model="llama3.2-vision",
    messages=[
        {
            "role": "user",
            "content": system_prompt,
            "image": [image_path],
        }
    ],
)
# print response
print(response["message"]["content"])
result = response["message"]["content"]
with open(os.path.join("responses", file_md), "w", encoding="utf_8") as file:
    file.write(result)

print("File saved sucessfully.")
