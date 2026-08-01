with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Remove duplicate block from line 412 to 613 (0-indexed 411 to 612)
del lines[411:613]

content = "".join(lines)

# Fix encoding artifacts
content = content.replace('"\\"', '"""')
content = content.replace('"\\",', '""",')
content = content.replace('??? Image Analysis', '🖼️ Image Analysis')
content = content.replace('?? Audio Analysis', '🔊 Audio Analysis')
content = content.replace('?? Comprehensive Diagnosis', '🧠 Comprehensive Diagnosis')
content = content.replace('??? Predicted Condition:', '🖼️ Predicted Condition:')
content = content.replace('?? Predicted Condition:', '🔊 Predicted Condition:')
content = content.replace('??? Image (X-ray / MRI)', '🖼️ Image (X-ray / MRI)')
content = content.replace('?? Text (Symptoms / Doctor Notes)', '📝 Text (Symptoms / Doctor Notes)')
content = content.replace('?? Audio (Breathing / Voice)', '🔊 Audio (Breathing / Voice)')


with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
