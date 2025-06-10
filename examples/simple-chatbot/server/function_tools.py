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

lookup_knowledge_base_function = FunctionSchema(
    name="lookup_knowledge_base",
    description="Looks up information in a knowledge base based on interest category, query, and optional tags.",
    properties={
        "query": {"type": "string", "description": "The query to search for"},
        "interest_category": {"type": "string", "description": "The category of interest, e.g., 'activities', 'legal_rights'"},
        "hobby_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of hobby tags to filter activities",
        },
        "location_tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional list of location tags to filter activities",
        },
    },
    required=["query", "interest_category"],
)

google_search_function = FunctionSchema(
    name="google_search",
    description="Performs a Google search to find information on the internet.",
    properties={
        "query": {"type": "string", "description": "The search query"},
    },
    required=["query"],
)

check_contact_exists_function = FunctionSchema(
    name="check_contact_exists",
    description="Checks if a contact exists in the user's phone by sending a message to the app.",
    properties={
        "contact_name": {"type": "string", "description": "The name of the contact to check"},
    },
    required=["contact_name"],
)