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