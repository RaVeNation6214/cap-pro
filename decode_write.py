import base64, os
data = "PLACEHOLDER"
content = base64.b64decode(data).decode("utf-8")
filepath = "C:/Users/aksha/OneDrive/Desktop/cap pro/frontend/src/pages/Results.jsx"
with open(filepath, "w", encoding="utf-8") as f:
    f.write(content)
size = os.path.getsize(filepath)
print("Done")
print(f"File size: {size} bytes")