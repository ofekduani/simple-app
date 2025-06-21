#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

"""Gemini Bot Implementation.

This module implements a chatbot using Google's Gemini Multimodal Live model.
It includes:
- Real-time audio/video interaction through Daily
- Animated robot avatar
- Speech-to-speech model

The bot runs as part of a pipeline that processes audio/video frames and manages
the conversation flow using Gemini's streaming capabilities.
"""

import asyncio
from datetime import date
import json
import os
import sys
import argparse

import aiohttp
from dotenv import load_dotenv
from loguru import logger
from PIL import Image
from runner import configure

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    OutputImageRawFrame,
    SpriteFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from profiles import PROFILES
from function_tools import (
    medicine_reminder_function,
    lookup_knowledge_base_function,
    google_search_function,
    check_contact_exists_function,
    manage_journal_takeaways_function # Added
)

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

current_user_id_for_handlers = None
KNOWLEDGE_BASE_FILE = "user_journal_takeaways.txt" # Added for journaling

# Load knowledge base data
activities_data = []
legal_rights_data = []
try:
    with open(os.path.join(os.path.dirname(__file__), "data/activities.json"), "r") as f:
        activities_data = json.load(f)
    with open(os.path.join(os.path.dirname(__file__), "data/legal_rights.json"), "r") as f:
        legal_rights_data = json.load(f)
except FileNotFoundError:
    logger.error("Knowledge base JSON files not found. Make sure they are in the server/data directory.")
except json.JSONDecodeError:
    logger.error("Error decoding JSON from knowledge base files.")


sprites = []
script_dir = os.path.dirname(__file__)

for i in range(1, 26):
    # Build the full path to the image file
    full_path = os.path.join(script_dir, f"assets/robot0{i}.png")
    # Get the filename without the extension to use as the dictionary key
    # Open the image and convert it to bytes
    with Image.open(full_path) as img:
        sprites.append(OutputImageRawFrame(image=img.tobytes(), size=img.size, format=img.format))

# Create a smooth animation by adding reversed frames
flipped = sprites[::-1]
sprites.extend(flipped)

# Define static and animated states
quiet_frame = sprites[0]  # Static frame for when bot is listening
talking_frame = SpriteFrame(images=sprites)  # Animation sequence for when bot is talking


class TalkingAnimation(FrameProcessor):
    """Manages the bot's visual animation states.

    Switches between static (listening) and animated (talking) states based on
    the bot's current speaking status.
    """

    def __init__(self):
        super().__init__()
        self._is_talking = False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process incoming frames and update animation state.

        Args:
            frame: The incoming frame to process
            direction: The direction of frame flow in the pipeline
        """
        await super().process_frame(frame, direction)

        # Switch to talking animation when bot starts speaking
        if isinstance(frame, BotStartedSpeakingFrame):
            if not self._is_talking:
                await self.push_frame(talking_frame)
                self._is_talking = True
        # Return to static frame when bot stops speaking
        elif isinstance(frame, BotStoppedSpeakingFrame):
            await self.push_frame(quiet_frame)
            self._is_talking = False

        await self.push_frame(frame, direction)


# Added: Function to load previous takeaways
async def load_previous_takeaways():
    try:
        with open(KNOWLEDGE_BASE_FILE, "r") as f:
            return f.read()
    except FileNotFoundError:
        return ""

# Modified: System prompt for journaling
def build_system_prompt(profile, previous_takeaways_text):
    # Determine user's name, defaulting to "me" if not available in profile
    user_name = profile.get('name', 'me') if profile else 'me'

    prompt = f"""
    You are my personal journaling assistant. Your goals are:
    1. Accurately transcribe my spoken journal entry.
    2. After I finish speaking and you have the full transcription, identify 2-3 key takeaways, insights, or main points from my entry.
    3. Present these key takeaways back to me.
    4. You will then use the 'manage_journal_takeaways' tool with the 'save' action to save these takeaways.

    You are speaking with {user_name}.
    Keep your interaction focused on these tasks. Be empathetic and understanding in tone when presenting takeaways.

    Here are some key takeaways from our previous sessions, for your context:
    {previous_takeaways_text if previous_takeaways_text else "This is our first session, or no takeaways were saved previously."}

    When I start speaking, listen carefully for my journal entry.
    Once I'm done, first present the key takeaways clearly.
    Example of presenting takeaways: "Okay, I've got that down. The key takeaways I noted from your entry are: 1. [Takeaway 1], 2. [Takeaway 2], and 3. [Takeaway 3]."
    After presenting the takeaways, you MUST call the 'manage_journal_takeaways' function with the 'save' action and the identified takeaways.
    """
    return prompt


async def main():
    """Main bot execution function.

    Sets up and runs the bot pipeline including:
    - Daily video transport with specific audio parameters
    - Gemini Live multimodal model integration
    - Voice activity detection
    - Animation processing
    - RTVI event handling
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--profile", dest="user_id", type=str, required=False, help="User profile ID (a or b)")
    args, unknown = parser.parse_known_args()
    user_id = args.user_id or os.getenv("USER_ID", "a")
    global current_user_id_for_handlers
    current_user_id_for_handlers = user_id
    profile = PROFILES.get(user_id, PROFILES["a"]) # Profile can still be used for user's name, etc.
    print(f"Loaded profile: {profile.get('name') if profile else 'default'}")

    previous_takeaways_text = await load_previous_takeaways() # Load at the beginning

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        # Set up Daily transport with specific audio/video parameters for Gemini
        transport = DailyTransport(
            room_url,
            token,
            "Chatbot",
            DailyParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                video_out_enabled=True,
                video_out_width=1024,
                video_out_height=576,
                vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.5)),
            ),
        )

        # Initialize the Gemini Multimodal Live model
        llm = GeminiMultimodalLiveLLMService(
            api_key=os.getenv("GEMINI_API_KEY"),
            voice_id="Puck",  # Aoede, Charon, Fenrir, Kore, Puck
            transcribe_user_audio=True,
            tools=ToolsSchema(standard_tools=[
                manage_journal_takeaways_function, # Added
                medicine_reminder_function, # Kept for potential other uses or future expansion
                lookup_knowledge_base_function, # Kept
                google_search_function, # Kept
                check_contact_exists_function # Kept
            ]),
        )

        llm.register_function("manage_journal_takeaways", handle_manage_journal_takeaways) # Added
        llm.register_function("set_medicine_reminder", set_medicine_reminder)
        llm.register_function("lookup_knowledge_base", lookup_knowledge_base)
        llm.register_function("google_search", google_search)
        llm.register_function("check_contact_exists", check_contact_exists)

        messages = [
            {
                "role": "user", # System prompt is passed as a user message to Gemini API in this setup
                "content": build_system_prompt(profile, previous_takeaways_text),
            },
            # Optional: Initial greeting from assistant to set the context
            # {
            #     "role": "assistant",
            #     "content": f"Hello {profile.get('name', 'there') if profile else 'there'}! Journaling session started. I've reviewed notes from our previous sessions."
            # }
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

        #
        # RTVI events for Pipecat client UI
        #
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        pipeline = Pipeline(
            [
                transport.input(),
                rtvi,
                context_aggregator.user(),
                llm,
                ta,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
                enable_usage_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )
        await task.queue_frame(quiet_frame)

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            # Kick off the conversation
            await task.queue_frames([context_aggregator.user().get_context_frame()])

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            print(f"Participant joined: {participant}")

        @transport.event_handler("on_participant_left")
        async def on_participant_left(transport, participant, reason):
            print(f"Participant left: {participant}")
            await task.cancel()

        runner = PipelineRunner()

        await runner.run(task)


# Added: Handler for manage_journal_takeaways function
async def handle_manage_journal_takeaways(params: FunctionCallParams):
    action = params.arguments.get("action")
    logger.info(f"Function call: manage_journal_takeaways, Action: {action}")

    if action == "save":
        takeaways = params.arguments.get("takeaways_to_save", [])
        if takeaways:
            try:
                with open(KNOWLEDGE_BASE_FILE, "a") as f:  # Append mode
                    f.write(f"Session Date: {date.today().isoformat()}\n")
                    for takeaway in takeaways:
                        f.write(f"- {takeaway}\n")
                    f.write("\n")  # Add a separator for readability
                logger.info(f"Saved takeaways: {takeaways}")
                await params.result_callback({"status": "success", "message": "Takeaways saved."})
            except Exception as e:
                logger.error(f"Error saving takeaways: {e}")
                await params.result_callback({"status": "error", "message": "Failed to save takeaways."})
        else:
            logger.warning("No takeaways provided to save.")
            await params.result_callback({"status": "no_action", "message": "No takeaways provided to save."})

    elif action == "load":
        # This action is primarily for completeness in the tool's definition.
        # Actual loading happens at the start of main() for context priming.
        # Gemini might theoretically call this if explicitly prompted by a user in a complex way.
        try:
            previous_takeaways_text = await load_previous_takeaways()
            await params.result_callback({"status": "success", "previous_takeaways": previous_takeaways_text})
        except Exception as e:
            logger.error(f"Error loading takeaways via function call: {e}")
            await params.result_callback({"status": "error", "message": "Failed to load takeaways."})
    else:
        logger.warning(f"Invalid action for manage_journal_takeaways: {action}")
        await params.result_callback({"status": "error", "message": "Invalid action specified."})


async def set_medicine_reminder(params: FunctionCallParams):
    logger.info(f"Function call received: set_medicine_reminder with params: {params.arguments}")
    global current_user_id_for_handlers
    if not current_user_id_for_handlers:
        logger.error("User ID not set for handler")
        await params.result_callback({"status": "failure", "error": "User ID not configured"})
        return

    profile = PROFILES.get(current_user_id_for_handlers)
    if not profile:
        logger.error(f"Profile not found for user ID: {current_user_id_for_handlers}")
        await params.result_callback({"status": "failure", "error": "Profile not found"})
        return

    today_str = date.today().isoformat()
    medicine_name = params.arguments.get("name")
    reminder_time = params.arguments.get("time")

    if profile.get("last_pill_taken_date") == today_str:
        logger.info(f"Medicine {medicine_name} already taken today by user {current_user_id_for_handlers}")
        await params.result_callback({
            "status": "already_taken",
            "medicine_name": medicine_name,
            "date": today_str
        })
    else:
        PROFILES[current_user_id_for_handlers]["last_pill_taken_date"] = today_str
        logger.info(f"Medicine reminder set for {medicine_name} at {reminder_time} for user {current_user_id_for_handlers}. Profile updated.")
        await params.result_callback({
            "status": "reminder_set",
            "medicine_name": medicine_name,
            "time": reminder_time,
            "date": today_str
        })


async def lookup_knowledge_base(params: FunctionCallParams):
    logger.info(f"Function call received: lookup_knowledge_base with params: {params.arguments}")
    query = params.arguments.get("query")
    interest_category = params.arguments.get("interest_category")
    hobby_tags = params.arguments.get("hobby_tags", [])
    location_tags = params.arguments.get("location_tags", [])

    results = []
    if interest_category == "activities":
        for activity in activities_data:
            matches_hobby = not hobby_tags or any(tag in activity.get("hobby_tags", []) for tag in hobby_tags)
            matches_location = not location_tags or any(tag in activity.get("location_tags", []) for tag in location_tags)
            matches_query = not query or query.lower() in activity.get("name", "").lower() or query.lower() in activity.get("description", "").lower()

            if matches_hobby and matches_location and (matches_query or (hobby_tags or location_tags)): # Match if tags are present or query matches
                results.append(activity)
    elif interest_category == "legal_rights":
        for right in legal_rights_data:
            matches_query = not query or query.lower() in right.get("topic", "").lower() or query.lower() in right.get("summary", "").lower()
            if matches_query:
                results.append(right)

    if results:
        # Format results for LLM
        formatted_results = []
        if interest_category == "activities":
            for res in results:
                formatted_results.append(f"- {res['name']}: {res['description']} (Hobbies: {', '.join(res.get('hobby_tags',[]))}, Location: {', '.join(res.get('location_tags',[]))})")
        elif interest_category == "legal_rights":
            for res in results:
                formatted_results.append(f"- {res['topic']}: {res['summary']} (More info: {res['details_prompt']})")
        await params.result_callback({"results": "\n".join(formatted_results)})
    else:
        await params.result_callback({"results": f"I couldn't find specific information for '{query}' in {interest_category}."})


async def google_search(params: FunctionCallParams):
    query = params.arguments.get("query")
    logger.info(f"Function call received: google_search with query: {query}")
    # In a real application, you would make an API call to a search engine here.
    # For this example, we'll just return a simulated result.
    simulated_result = f"Simulated Google Search results for '{query}': The capital of France is Paris. More information can be found online."
    await params.result_callback({"results": simulated_result})


async def check_contact_exists(params: FunctionCallParams):
    contact_name = params.arguments.get("contact_name")
    logger.info(f"Function call received: check_contact_exists for contact_name: {contact_name}")
    # This is where you would typically send an RTVI message to the client application.
    # For example: await rtvi.send_message_to_client({"action": "check_contact", "name": contact_name})
    # For this subtask, we are just logging and returning a guiding message to the LLM.
    logger.info(f"Simulating RTVI message for contact check: {{\"action\": \"check_contact\", \"name\": \"{contact_name}\"}}")
    await params.result_callback({
        "status": "checking_contact",
        "contact_name": contact_name,
        "message": "I've asked the app to check for this contact. The app will show the result if found."
    })


if __name__ == "__main__":
    asyncio.run(main())
