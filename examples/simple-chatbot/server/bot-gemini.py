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
import os
import sys
import argparse
import functools # Added for Phase 6

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
import json # Added for lookup_doc
from pipecat.services.gemini_multimodal_live.gemini import GeminiMultimodalLiveLLMService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.services.daily import DailyParams, DailyTransport
from profiles import PROFILES
from function_tools import medicine_reminder_function, lookup_doc_function, set_phone_alarm_function # Added set_phone_alarm_function

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

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


def build_system_prompt(profile):
    return f"""
    You are a helpful assistant for {profile.get('name', 'the user')}.
    You MUST conduct the entire conversation in Hebrew. Please respond and interact with the user exclusively in Hebrew.
    They are {profile.get('age')} years old, live in {profile.get('city')}, and enjoy {', '.join(profile.get('hobbies', []))}.
    Their pill time is {profile.get('pill_time')}.
    Current status for pill_taken_today: {profile.get('pill_taken_today', False)}.

    You have access to a tool called `set_medicine_reminder`. Use this tool when the user asks you to set a medicine reminder.
    When the user asks to set a reminder, you need to identify the medicine name and the time for the reminder.
    If the user's request is missing either the medicine name or the time, you MUST ask the user for the missing information before calling the tool.
    Ask for the missing information in Hebrew.
    For example, if the medicine name is missing, ask: #[Hebrew: "What is the name of the medicine?"]#
    If the time is missing, ask: #[Hebrew: "At what time should I set the reminder?"]#
    Only call the `set_medicine_reminder` function AFTER you have successfully gathered both the medicine name and the time from the user.
    When you set a medicine reminder, the system also notes that the user's pill for the day has been accounted for (`pill_taken_today` will be true). If the user asks to set a reminder and you find `pill_taken_today` is already true in their profile information (available to you in this system prompt), you can inform them, for example: 'I've set the reminder for [medicine] at [time]. I also see you've already noted taking your medication today.'

    If the user asks if they've taken their medicine today, check the `pill_taken_today` status in their profile and answer accordingly in Hebrew. For example, if true: 'כן, לפי המידע שלי כבר לקחת את התרופה שלך היום.' (Yes, according to my information you have already taken your medicine today.) If false: 'לא רשום שלקחת את התרופה שלך היום. תרצה שאזכיר לך?' (It's not noted that you've taken your medicine today. Would you like me to remind you?)

    You also have a tool called `lookup_doc`. Use this tool if the user asks about activities, schedules, or any other information that might be found in a local knowledge base. Pass the user's question or key topics as the 'query' to the tool. For example, if the user asks "מה יש בימי שני אחר הצהריים?", you should call `lookup_doc` with a query like "ימי שני" or "שני אחר הצהריים".

    You have a tool called `set_phone_alarm` to set an alarm on the user's phone. If the user asks to set an alarm or wake-up call, use this tool. You must get the specific time in HH:MM format. An alarm label is optional. For example, if the user says 'Set an alarm for 7 AM to take my medicine', call `set_phone_alarm` with `time` as '07:00' and `label` as 'Take medicine'.
    """


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
    profile = PROFILES.get(user_id, PROFILES["a"])
    print(f"Loaded profile: {profile.get('name')}")

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
            tools=ToolsSchema(standard_tools=[medicine_reminder_function, lookup_doc_function, set_phone_alarm_function]), # Added set_phone_alarm_function
        )

        # RTVI events for Pipecat client UI
        # This needs to be initialized before registering functions that use it.
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # Register function handlers
        llm.register_function("set_medicine_reminder", functools.partial(set_medicine_reminder_handler, profile=profile))
        llm.register_function("lookup_doc", lookup_doc)
        llm.register_function("set_phone_alarm", functools.partial(set_phone_alarm_handler, rtvi=rtvi)) # Added set_phone_alarm handler

        messages = [
            {
                "role": "user",
                "content": build_system_prompt(profile),
            },
        ]

        # Set up conversation context and management
        # The context_aggregator will automatically collect conversation context
        context = OpenAILLMContext(messages)
        context_aggregator = llm.create_context_aggregator(context)

        ta = TalkingAnimation()

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


async def set_medicine_reminder_handler(params: FunctionCallParams, profile: dict):
    logger.info(f"Function call received: set_medicine_reminder with params: {params.arguments} for profile: {profile.get('name')}")

    # Update pill_taken_today status
    profile["pill_taken_today"] = True
    logger.info(f"Profile for {profile.get('name')}: pill_taken_today set to True")

    await params.result_callback({
        "status": "Reminder set and pill_taken_today flag updated to True.",
        "medicine_name": params.arguments.get("name"),
        "time": params.arguments.get("time"),
        "pill_taken_today": True
    })

async def lookup_doc(params: FunctionCallParams):
    query = params.arguments.get("query")
    logger.info(f"Function call received: lookup_doc with query: {query}")

    found_info_string = "מצטער, לא מצאתי מידע על זה." # Default sorry message in Hebrew

    if query:
        try:
            # Construct the full path to knowledge.json relative to this script's directory
            knowledge_file_path = os.path.join(os.path.dirname(__file__), "data", "knowledge.json")
            with open(knowledge_file_path, 'r', encoding='utf-8') as f:
                knowledge_base = json.load(f)

            activities = knowledge_base.get("activities", [])
            found_activities = []

            for activity in activities:
                if query.lower() in activity.get("name", "").lower() or \
                   query.lower() in activity.get("description", "").lower() or \
                   query.lower() in activity.get("time", "").lower(): # also check time field
                    found_activities.append(f"פעילות: {activity.get('name', '')}, תיאור: {activity.get('description', '')}, זמן: {activity.get('time', '')}")

            if found_activities:
                found_info_string = "מצאתי את הפעילויות הבאות: " + " | ".join(found_activities)
        except FileNotFoundError:
            logger.error("knowledge.json not found at expected path.")
            found_info_string = "מצטער, קובץ המידע אינו זמין כרגע."
        except json.JSONDecodeError:
            logger.error("Error decoding knowledge.json.")
            found_info_string = "מצטער, יש בעיה בקריאת קובץ המידע."
        except Exception as e:
            logger.error(f"An unexpected error occurred during lookup_doc: {e}")
            found_info_string = "מצטער, אירעה שגיאה בלתי צפויה בעת חיפוש המידע."

    logger.info(f"lookup_doc result: {found_info_string}")
    await params.result_callback({"content": found_info_string})

async def set_phone_alarm_handler(params: FunctionCallParams, rtvi: RTVIProcessor):
    time = params.arguments.get("time")
    label = params.arguments.get("label", "") # Optional, defaults to empty string if not provided

    payload = {
        "action": "set_alarm",
        "time": time,
        "label": label
    }

    if rtvi:
        await rtvi.send_message(payload)
        logger.info(f"Sent RTVI message to client: set_alarm with payload: {payload}")
        await params.result_callback({"status": f"Attempted to send 'set_alarm' instruction to client for {time}.", "sent_payload": payload})
    else:
        logger.error("RTVIProcessor instance is not available. Cannot send set_alarm message.")
        await params.result_callback({"status": "Failed to send 'set_alarm' instruction: RTVI system not available.", "sent_payload": payload})


if __name__ == "__main__":
    asyncio.run(main())
