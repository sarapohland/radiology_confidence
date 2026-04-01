PROMPT="""
You are a helpful assistant. Please generate a report for the given images, including both findings and impressions. 

For context on the patient information, examination details, clinical history, and indication use the following information: {context_info}

Return the report in the following format: Findings: {{}} Impression: {{}}.

Always format your output in the EXACT following way:\n"
    "[Report]: \nInsert report here.\n\n"
"""