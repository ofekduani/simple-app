# Simple-Chatbot Project: Phase 2, Step 2.5 Test Plan

**Objective:** Test the `set_medicine_reminder` function calling setup by sending sample queries to the server using curl and observing the server logs.

**Revised Plan:**

1.  **User Action Required**: Ensure the server is running by running `python3 examples/simple-chatbot/server/server.py` in a terminal. Confirm the ngrok forwarding address (e.g., `https://4632-141-226-93-22.ngrok-free.app`).
2.  **Establish Connection**: Execute a curl command to establish a connection with the server using the `/connect` endpoint and the user ID.
    ```bash
    curl -v -X POST "https://4632-141-226-93-22.ngrok-free.app/connect?user_id=a"
    ```
    *(Note: The `-v` flag provides verbose output, which might be helpful for observing the connection process.)*
3.  **Simulate Complete Request**: Once the connection is established (or the `/connect` command completes), execute a curl command to simulate a user asking to set a medicine reminder with all necessary information (medicine name and time) using the `/api/offer` endpoint.
    ```bash
    curl -X POST "https://4632-141-226-93-22.ngrok-free.app/api/offer" -H "Content-Type: application/json" -d '{"sdp": "...", "type": "offer", "pc_id": "test_user_a", "user_id": "a", "text": "תזכיר לי לקחת אדוויל בשעה שמונה בבוקר"}'
    ```
4.  **Validate Complete Request**: Check the server logs for output from the `set_medicine_reminder` handler. Look for a log message similar to "Function call received: set\_medicine\_reminder with params: {'name': 'אדוויל', 'time': '08:00'}".
5.  **Simulate Incomplete Request (Missing Name)**: Execute a curl command to simulate a user asking to set a reminder with only the time, using the `/api/offer` endpoint.
    ```bash
    curl -X POST "https://4632-141-226-93-22.ngrok-free.app/api/offer" -H "Content-Type: application/json" -d '{"sdp": "...", "type": "offer", "pc_id": "test_user_b", "user_id": "a", "text": "תזכיר לי לקחת תרופה בשעה שמונה בערב"}'
    ```
6.  **Validate Incomplete Request (Missing Name)**: Check the server logs. The LLM should *not* call the function. Instead, it should generate a response asking for the missing information (the medicine name) in Hebrew.
7.  **Simulate Incomplete Request (Missing Time)**: Execute a curl command to simulate a user asking to set a reminder with only the medicine name, using the `/api/offer` endpoint.
    ```bash
    curl -X POST "https://4632-141-226-93-22.ngrok-free.app/api/offer" -H "Content-Type: application/json" -d '{"sdp": "...", "type": "offer", "pc_id": "test_user_c", "user_id": "a", "text": "תזכיר לי לקחת טאמס"}'
    ```
8.  **Validate Incomplete Request (Missing Time)**: Check the server logs. The LLM should *not* call the function. Instead, it should generate a response asking for the missing information (the time) in Hebrew.

**Validation:**
Confirm with me that you have performed the steps and observed the expected output in the server logs for each curl command, including the function call with parameters for the complete request and Hebrew questions for the incomplete requests.

**Constraint:**
Only perform testing via curl and log observation. Do not modify code in this step.

**Completion:**
Signal completion by using the `attempt_completion` tool, providing a concise summary of the testing performed and the validation results.