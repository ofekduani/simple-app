from pipecat.adapters.schemas.function_schema import FunctionSchema

medicine_reminder_function = FunctionSchema(
    name="set_medicine_reminder",
    description="Sets a medicine reminder for the user.",
    properties={
        "name": {"type": "string", "description": "The name of the medicine"},
        "time": {"type": "string", "description": "The time to set the reminder for, in HH:MM format"},
    },
    required=["name", "time"],
)

lookup_doc_function = FunctionSchema(
    name="lookup_doc",
    description="Looks up information from a knowledge base about activities or other topics. Use this if the user asks a question that might be answered by looking up specific information.",
    properties={
        "query": {"type": "string", "description": "The user's query or keywords to search for in the knowledge base. This should be in Hebrew if the user spoke Hebrew."}
    },
    required=["query"],
)

set_phone_alarm_function = FunctionSchema(
    name="set_phone_alarm",
    description="Sets a native phone alarm on the user's device. Use this if the user explicitly asks to set an alarm or wake-up call.",
    properties={
        "time": {"type": "string", "description": "The time to set the alarm for, in HH:MM format (24-hour). For example, '08:00' or '21:30'."},
        "label": {"type": "string", "description": "An optional label for the alarm. For example, 'Morning Exercise' or 'Pill Reminder'.", "optional": True}
    },
    required=["time"],
)