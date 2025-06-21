# Personal Journaling App MVP - Setup and Usage Guide

## 1. Overview

This document provides instructions to set up and run the Personal Journaling App MVP. This application allows you to speak your journal entries, have them transcribed, and have the AI identify and save key takeaways from each session. These takeaways are used to provide context in future journaling sessions.

**Core MVP Features:**
*   Voice-based journal entries.
*   Real-time transcription of your speech.
*   AI-powered identification of 2-3 key takeaways from your journal entry.
*   Persistent storage of these key takeaways in a local file (`user_journal_takeaways.txt`) on the server.
*   Contextual awareness: previous takeaways are loaded at the start of new sessions to inform the AI.

This MVP uses a modified version of the `simple-chatbot` example, leveraging its Gemini integration for STT, LLM, and TTS capabilities.

## 2. Server Setup

The backend server processes your voice, interacts with the Gemini AI, and manages the takeaways knowledge base.

**Prerequisites:**
*   Python 3.10 or newer.
*   `git` (if cloning the repository).

**Steps:**

1.  **Get the Code:**
    *   If you haven't already, clone the repository containing the `pipecat-python` examples.
    *   Navigate to the server directory:
        ```bash
        cd path/to/pipecat-python/examples/simple-chatbot/server/
        ```

2.  **Create a Python Virtual Environment:**
    *   It's highly recommended to use a virtual environment to manage dependencies.
        ```bash
        python3 -m venv venv
        ```
    *   Activate the virtual environment:
        *   On macOS and Linux:
            ```bash
            source venv/bin/activate
            ```
        *   On Windows:
            ```bash
            venv\\Scripts\\activate
            ```

3.  **Install Dependencies:**
    *   Install the required Python packages:
        ```bash
        pip install -r requirements.txt
        ```

4.  **Configure API Keys and Settings (`.env` file):**
    *   In the `examples/simple-chatbot/server/` directory, find the `env.example` file.
    *   Create a copy of this file and name it `.env`:
        ```bash
        cp env.example .env
        ```
    *   Open the `.env` file in a text editor and fill in the necessary API keys and settings:
        *   **`GEMINI_API_KEY`**: **Required.** Your API key for Google Gemini services. You can obtain this from Google AI Studio.
        *   **`DAILY_API_KEY`**: **Required.** Your API key for Daily.co, used for real-time audio/video transport. You can get this from the Daily.co dashboard.
        *   **`BOT_IMPLEMENTATION`**: Ensure this is set to `gemini`.
            ```ini
            BOT_IMPLEMENTATION=gemini
            ```
        *   **Other API Keys (Optional for this MVP):** The `env.example` file lists keys for other services (OpenAI, ElevenLabs, etc.). These are **not required** for this specific Gemini-based journaling MVP. You can leave them blank or remove them.
        *   **`USER_ID` (Optional):** You can set a default `USER_ID` (e.g., `USER_ID=a`) if you want to consistently use a specific profile from `profiles.py`. The journaling app primarily uses the profile for the user's name in prompts.

5.  **Run the Server:**
    *   Once the `.env` file is configured, start the server from the `examples/simple-chatbot/server/` directory:
        ```bash
        python server.py
        ```
        This command runs the FastAPI server (`server.py`), which in turn manages and runs the `bot-gemini.py` logic when a client connects via the `/connect` endpoint.
    *   You should see log output indicating the FastAPI server is running (e.g., on `http://localhost:7860`) and then further logs from `bot-gemini.py` when a client connects.

6.  **Using `ngrok` (Optional - for external access):**
    *   If your Android device/emulator is not on the same local network as your server, you'll need a tool like `ngrok` to expose your local server (specifically, the `server.py` FastAPI application) to the internet.
    *   The `server.py` application typically runs on `localhost:7860`. To expose this, use:
        ```bash
        ngrok http 7860
        ```
    *   `ngrok` will provide you with a public HTTPS URL (e.g., `https://xxxx-yyy-zzz.ngrok.io`). You will use this public URL (including the `/connect` path, e.g., `https://xxxx-yyy-zzz.ngrok.io/connect`) in the Android client's server URL configuration.

## 3. Android Client Setup

The Android client allows you to interact with the journaling bot.

**Prerequisites:**
*   Android Studio (latest stable version recommended).
*   Android SDK installed and configured.
*   An Android Emulator or a physical Android device (with USB debugging enabled).

**Steps:**

1.  **Open the Project in Android Studio:**
    *   Launch Android Studio.
    *   Select "Open" or "Open an Existing Project."
    *   Navigate to and select the `examples/simple-chatbot/client/android/` directory. Android Studio should recognize it as a valid project.

2.  **Ensure Server is Running and Accessible:**
    *   Your Python server (from Section 2) must be running.
    *   The Android client needs to be able to connect to the server.
        *   **Same Network:** If your Android device/emulator and your server are on the same Wi-Fi network, you can usually use your computer's local IP address for the server URL in the client.
        *   **Emulator to Localhost:** If using an Android emulator on the same machine as the server, `10.0.2.2` is typically the alias for your computer's `localhost`. So, if the server's `/connect` endpoint is `http://localhost:7860`, you'd use `http://10.0.2.2:7860` in the client.
        *   **Different Networks:** If using `ngrok`, use the public `https://` URL provided by `ngrok` in the client.

3.  **Configure Server URL in Android Client (If Necessary):**
    *   The Android client needs to know the address of your server's `/connect` endpoint (which is managed by `server.py` that then runs `bot-gemini.py`).
    *   Check the Android client's code for where the server URL is defined. This is often in a constants file (e.g., `Constants.kt`) or a settings/configuration screen within the app.
    *   For this MVP, if it's hardcoded, you might need to temporarily change it in the Kotlin code to point to your server's address (e.g., `private const val BASE_URL = "http://10.0.2.2:7860"`).
    *   **Default for `simple-chatbot`:** The client is often configured to connect to a server URL that's either hardcoded or can be input in a settings screen. Check the client's `README.md` or code.

4.  **Build and Run the App:**
    *   In Android Studio, select your target device (emulator or physical device).
    *   Click the "Run" button (green play icon) or use the menu `Run > Run 'app'`.
    *   Android Studio will build the project and install it on the selected device/emulator. The app should launch automatically.

## 4. How to Use the Journaling MVP

1.  **Start the Server:** Make sure your Python server (`bot-gemini.py` via `server.py` ideally) is running.
2.  **Launch the Android App:** Open the "Simple Chatbot" app on your Android device/emulator.
3.  **Connect:**
    *   Use the app's interface to connect to the server. You might need to select a profile (e.g., "User A" or "User B" – the profile itself isn't critical for journaling MVP beyond the name, but the connection needs to be established).
4.  **Speak Your Journal Entry:**
    *   Once connected, the bot (after its initial greeting, if any) will be listening.
    *   Speak clearly into your device's microphone. For example: "Today, I worked on setting up the new project environment. It was challenging but I managed to get the server running. I also had a good conversation with a colleague about future plans."
5.  **Observe the Interaction:**
    *   You should see your speech being transcribed in the app's chat interface (as "user" messages).
    *   The bot will then respond. Its response should:
        1.  Acknowledge your entry.
        2.  State the 2-3 key takeaways it identified from your speech. For the example above, it might say: "Okay, I've got that down. The key takeaways I noted from your entry are: 1. You worked on setting up a new project environment, 2. It was challenging but you succeeded in getting the server running, and 3. You had a positive conversation about future plans."
6.  **Check Saved Takeaways (Server-Side):**
    *   After the bot states the takeaways, it will use the `manage_journal_takeaways` tool to save them.
    *   Look in the `examples/simple-chatbot/server/` directory for a file named `user_journal_takeaways.txt`.
    *   Open it. You should see the takeaways from your session, prefixed with the date.
7.  **Subsequent Sessions:**
    *   Close and reopen the app (or disconnect and reconnect).
    *   When you start a new session, the server will load the content of `user_journal_takeaways.txt`.
    *   The system prompt given to Gemini will now include these previous takeaways, allowing the bot to have more context about your journaling history. You can verify this by checking the server logs for the content of the system prompt if you add logging for it.

## 5. Troubleshooting Tips

*   **Server Not Starting:**
    *   Check Python version.
    *   Ensure virtual environment is activated.
    *   Verify all dependencies in `requirements.txt` are installed.
    *   Make sure your `.env` file is correctly named and placed, and that `GEMINI_API_KEY` and `DAILY_API_KEY` are valid.
    *   Ensure the server port (e.g., 7860) is not already in use by another application.
*   **Android Client Not Connecting:**
    *   Confirm the server (`server.py`) is running.
    *   Double-check the server URL configured in the Android client. Use `10.0.2.2` for localhost if using an emulator on the same machine as the server.
    *   Check network connectivity for both server and client. If on different networks, ensure `ngrok` (or similar) is set up correctly for the signaling server endpoint.
    *   Look at Android Studio's Logcat for error messages from the client.
    *   Check server logs for connection attempts or errors.
*   **No Transcription / Bot Not Responding:**
    *   Ensure microphone permissions are granted for the app on your Android device.
    *   Verify API keys are correct and have not exceeded quotas.
    *   Check server logs for any errors from the Gemini service or the Daily transport.
*   **Takeaways Not Saving:**
    *   Check server logs for errors related to file operations (`user_journal_takeaways.txt`).
    *   Ensure the server has write permissions in the `examples/simple-chatbot/server/` directory.

This guide should help you get the journaling app MVP up and running!
