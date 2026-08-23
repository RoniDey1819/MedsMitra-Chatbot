/**
 * Medical Shop Chatbot Widget
 * Drop this into any HTML page:
 *
 *   <script>
 *     window.MED_CHATBOT_API_URL = "https://your-backend.onrender.com/chat";
 *     window.MED_CHATBOT_ICON_URL = "https://yoursite.com/bot-icon.png"; // optional, .png/.gif/.webp
 *     window.MED_CHATBOT_GREETING_TEXT = "Say hi to Med! 👋"; // optional, defaults shown
 *     window.MED_CHATBOT_SHOW_GREETING = true;  // optional, set false to disable the popup
 *     window.MED_CHATBOT_GREETING_DELAY_MS = 1500; // optional, delay before it appears
 *   </script>
 *   <script src="chatbot-widget.js"></script>
 *
 * That's it - a floating chat bubble appears in the bottom-right corner.
 * Set MED_CHATBOT_ICON_URL to use a custom image/gif instead of the default
 * 💬 emoji; when a custom icon is set, the circular background is removed
 * so only the icon graphic shows (no colored disc behind it). A small
 * "Say hi to Med!" speech-bubble also pops up near the icon once per
 * browser (dismissible, and remembers the dismissal via localStorage).
 * The backend must expose:
 *   POST   {API_URL}                  (SSE stream: "data: {token|error|done}\n\n")
 *   GET    {API_URL}/{session_id}/history
 *   DELETE {API_URL}/{session_id}
 */
(function () {
  const CHAT_URL = window.MED_CHATBOT_API_URL || "http://localhost:8000/chat";
  const ICON_URL = window.MED_CHATBOT_ICON_URL || null; // e.g. "https://yoursite.com/bot-icon.png" or a .gif
  const SESSION_STORAGE_KEY = "medsmitra_session_id";

  // Voice search (mic button next to the input). Defaults to CHAT_URL's
  // origin + /transcribe, so it normally needs no separate config.
  const TRANSCRIBE_URL =
    window.MED_CHATBOT_TRANSCRIBE_URL ||
    CHAT_URL.replace(/\/chat\/?$/, "") + "/transcribe";
  const SHOW_VOICE_SEARCH = window.MED_CHATBOT_SHOW_VOICE_SEARCH !== false; // default: on

  // Greeting speech-bubble that pops up near the icon to invite a click.
  const GREETING_TEXT = window.MED_CHATBOT_GREETING_TEXT || "Say hi to Med! 👋";
  const SHOW_GREETING = window.MED_CHATBOT_SHOW_GREETING !== false; // default: on
  const GREETING_DELAY_MS = window.MED_CHATBOT_GREETING_DELAY_MS ?? 1500;
  const GREETING_DISMISSED_KEY = "medsmitra_greeting_dismissed";

  // ---------------------------------------------------------------
  // Session id - persisted in localStorage so a page reload resumes
  // the same conversation (backend keeps history in Redis for
  // SESSION_TTL_SECONDS after the last message).
  // ---------------------------------------------------------------

  function generateUuid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      const v = c === "x" ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function newSessionId() {
    const id = generateUuid();
    localStorage.setItem(SESSION_STORAGE_KEY, id);
    return id;
  }

  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY) || newSessionId();
  let historyRestored = false;

  // ---------------------------------------------------------------
  // Styles
  // ---------------------------------------------------------------

  const style = document.createElement("style");
  style.textContent = `
    #mc-bubble {
      position: fixed; bottom: 20px; right: 20px; width: 108px; height: 108px;
      border-radius: 50%; background: #0f766e; color: #fff; display: flex;
      align-items: center; justify-content: center; cursor: pointer;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25); font-size: 26px; z-index: 999999;
      border: none; transition: transform 0.15s ease; padding: 0;
    }
    #mc-bubble:hover { transform: scale(1.06); }
    #mc-bubble.mc-has-icon {
      background: transparent; box-shadow: none;
      filter: drop-shadow(0 4px 10px rgba(0,0,0,0.2));
    }
    #mc-bubble img { width: 100%; height: 100%; object-fit: contain; }
    #mc-greeting {
      position: fixed; bottom: 92px; right: 16px; max-width: 220px;
      background: #fff; color: #222; padding: 10px 32px 10px 14px; border-radius: 14px;
      box-shadow: 0 6px 20px rgba(0,0,0,0.18); font-size: 13.5px; line-height: 1.4;
      font-family: system-ui, -apple-system, sans-serif; z-index: 999998; cursor: pointer;
      opacity: 0; transform: translateY(6px) scale(0.96); pointer-events: none;
      transition: opacity 0.25s ease, transform 0.25s ease;
    }
    #mc-greeting.mc-visible { opacity: 1; transform: translateY(0) scale(1); pointer-events: auto; }
    #mc-greeting::after {
      content: ""; position: absolute; bottom: -7px; right: 26px;
      width: 12px; height: 12px; background: #fff;
      clip-path: polygon(0 0, 100% 0, 0 100%); transform: rotate(-45deg);
      box-shadow: 3px 3px 6px rgba(0,0,0,0.06);
    }
    #mc-greeting-close {
      position: absolute; top: 4px; right: 6px; background: none; border: none;
      color: #999; font-size: 14px; cursor: pointer; padding: 4px; line-height: 1;
    }
    #mc-greeting-close:hover { color: #555; }
    #mc-window {
      position: fixed; bottom: 90px; right: 20px; width: 340px; max-width: 90vw;
      height: 480px; max-height: 72vh; background: #fff; border-radius: 12px;
      box-shadow: 0 8px 30px rgba(0,0,0,0.25); display: none; flex-direction: column;
      overflow: hidden; z-index: 999999; font-family: system-ui, -apple-system, sans-serif;
    }
    #mc-header {
      background: #0f766e; color: #fff; padding: 12px 16px; font-weight: 600;
      display: flex; justify-content: space-between; align-items: center; font-size: 14px;
    }
    #mc-header-actions { display: flex; align-items: center; gap: 12px; }
    #mc-clear {
      cursor: pointer; background: none; border: none; color: #d1fae5;
      font-size: 12px; text-decoration: underline; padding: 0;
    }
    #mc-clear:disabled { opacity: 0.5; cursor: default; text-decoration: none; }
    #mc-close { cursor: pointer; background: none; border: none; color: #fff; font-size: 18px; padding: 0; }
    #mc-status { font-size: 11px; color: #b45309; background: #fffbeb; padding: 4px 12px; display: none; }
    #mc-messages { flex: 1; overflow-y: auto; padding: 12px; background: #f7f7f7; scroll-behavior: smooth; }
    .mc-msg { margin-bottom: 10px; max-width: 85%; padding: 8px 12px; border-radius: 10px; font-size: 14px; line-height: 1.4; }
    .mc-user { background: #0f766e; color: #fff; margin-left: auto; border-bottom-right-radius: 2px; }
    .mc-bot { background: #e9e9e9; color: #222; margin-right: auto; border-bottom-left-radius: 2px; white-space: pre-wrap; }
    .mc-dot { display: inline-block; width: 6px; height: 6px; margin: 0 2px; background: #888; border-radius: 50%; animation: mc-bounce 1s infinite ease-in-out; }
    .mc-dot:nth-child(2) { animation-delay: 0.15s; }
    .mc-dot:nth-child(3) { animation-delay: 0.3s; }
    @keyframes mc-bounce { 0%, 80%, 100% { transform: translateY(0); } 40% { transform: translateY(-4px); } }
    .mc-retry-btn {
      margin-top: 6px; display: inline-block; background: #0f766e; color: #fff;
      border: none; border-radius: 6px; padding: 4px 10px; font-size: 12px; cursor: pointer;
    }
    #mc-input-row { display: flex; align-items: center; border-top: 1px solid #ddd; }
    #mc-input { flex: 1; border: none; padding: 12px; font-size: 14px; outline: none; min-width: 0; }
    #mc-input:disabled { background: #f3f3f3; }
    #mc-mic {
      display: inline-flex; align-items: center; justify-content: center;
      width: 34px; height: 34px; margin: 0 2px; padding: 0; flex-shrink: 0;
      border-radius: 50%; border: none; background: transparent; color: #666;
      cursor: pointer; transition: background 0.15s ease, color 0.15s ease;
    }
    #mc-mic:hover { background: #eee; }
    #mc-mic.mc-recording { background: #fde2e2; color: #d33; }
    #mc-mic:disabled { opacity: 0.5; cursor: default; background: transparent; }
    #mc-send { background: #0f766e; color: #fff; border: none; padding: 0 16px; cursor: pointer; font-size: 14px; }
    #mc-send:disabled { opacity: 0.5; cursor: default; }
    .mc-disclaimer { font-size: 11px; color: #888; padding: 6px 12px; text-align: center; }
  `;
  document.head.appendChild(style);

  // ---------------------------------------------------------------
  // Markup
  // ---------------------------------------------------------------

  const bubble = document.createElement("button");
  bubble.id = "mc-bubble";
  if (ICON_URL) {
    bubble.classList.add("mc-has-icon");
    const iconImg = document.createElement("img");
    iconImg.src = ICON_URL;
    iconImg.alt = "Open chat";
    // If the custom image fails to load (bad URL, 404, etc.), fall back
    // to the emoji so the bubble is never left blank.
    iconImg.addEventListener("error", () => {
      bubble.classList.remove("mc-has-icon");
      bubble.innerHTML = "💬";
    });
    bubble.appendChild(iconImg);
  } else {
    bubble.innerHTML = "💬";
  }
  document.body.appendChild(bubble);

  // Greeting speech bubble - dismissible, shown once per browser.
  let greetingEl = null;
  if (SHOW_GREETING && !localStorage.getItem(GREETING_DISMISSED_KEY)) {
    greetingEl = document.createElement("div");
    greetingEl.id = "mc-greeting";
    greetingEl.innerHTML = `${GREETING_TEXT}<button id="mc-greeting-close" title="Dismiss" aria-label="Dismiss">✕</button>`;
    document.body.appendChild(greetingEl);

    const dismissGreeting = () => {
      greetingEl.classList.remove("mc-visible");
      localStorage.setItem(GREETING_DISMISSED_KEY, "1");
    };

    greetingEl
      .querySelector("#mc-greeting-close")
      .addEventListener("click", (e) => {
        e.stopPropagation();
        dismissGreeting();
      });
    greetingEl.addEventListener("click", () => {
      dismissGreeting();
      bubble.click();
    });

    setTimeout(() => greetingEl.classList.add("mc-visible"), GREETING_DELAY_MS);
  }

  const win = document.createElement("div");
  win.id = "mc-window";
  win.innerHTML = `
    <div id="mc-header">
      <span>Pharmacy Assistant</span>
      <div id="mc-header-actions">
        <button id="mc-clear" title="Clear conversation">Clear</button>
        <button id="mc-close" title="Close">✕</button>
      </div>
    </div>
    <div id="mc-status"></div>
    <div id="mc-messages"></div>
    <div class="mc-disclaimer">Informational only. Please confirm with our pharmacist.</div>
    <div id="mc-input-row">
      <input id="mc-input" type="text" placeholder="Ask about a medicine..." />
      <button id="mc-mic" type="button" title="Voice search" aria-label="Voice search">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
          <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
          <line x1="12" y1="19" x2="12" y2="23"></line>
          <line x1="8" y1="23" x2="16" y2="23"></line>
        </svg>
      </button>
      <button id="mc-send">Send</button>
    </div>
  `;
  document.body.appendChild(win);

  const messagesEl = win.querySelector("#mc-messages");
  const inputEl = win.querySelector("#mc-input");
  const sendBtn = win.querySelector("#mc-send");
  const clearBtn = win.querySelector("#mc-clear");
  const statusEl = win.querySelector("#mc-status");
  const micBtn = win.querySelector("#mc-mic");

  function showStatus(text) {
    statusEl.textContent = text;
    statusEl.style.display = text ? "block" : "none";
  }

  function addMessage(text, who) {
    const div = document.createElement("div");
    div.className = "mc-msg " + (who === "user" ? "mc-user" : "mc-bot");
    div.textContent = text;
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  function showTyping() {
    const div = document.createElement("div");
    div.className = "mc-msg mc-bot";
    div.innerHTML =
      '<span class="mc-dot"></span><span class="mc-dot"></span><span class="mc-dot"></span>';
    messagesEl.appendChild(div);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    return div;
  }

  // ---------------------------------------------------------------
  // Restore prior conversation (persistent session) on first open
  // ---------------------------------------------------------------

  async function restoreHistory() {
    try {
      const res = await fetch(`${CHAT_URL}/${sessionId}/history`);
      if (!res.ok) return;
      const data = await res.json();
      (data.history || []).forEach((turn) => {
        if (turn.role === "user") addMessage(turn.content, "user");
        else if (turn.role === "assistant") addMessage(turn.content, "bot");
      });
    } catch (err) {
      console.error("Failed to restore conversation history:", err);
    }
  }

  // ---------------------------------------------------------------
  // Send message - SSE streaming with reconnect/retry handling
  // ---------------------------------------------------------------

  async function sendMessage() {
    const text = inputEl.value.trim();
    if (!text) return;

    addMessage(text, "user");
    inputEl.value = "";
    sendBtn.disabled = true;
    inputEl.disabled = true;
    showStatus("");

    const botDiv = showTyping();
    let received = "";
    let gotFirstToken = false;

    try {
      const res = await fetch(CHAT_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }),
      });
      if (!res.ok || !res.body)
        throw new Error("Request failed: " + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // last part may be incomplete - keep for next chunk

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data:")) continue;

          let payload;
          try {
            payload = JSON.parse(line.slice(5).trim());
          } catch {
            continue;
          }

          if (payload.token) {
            if (!gotFirstToken) {
              botDiv.innerHTML = "";
              gotFirstToken = true;
            }
            received += payload.token;
            botDiv.textContent = received;
            messagesEl.scrollTop = messagesEl.scrollHeight;
          }

          if (payload.error) {
            botDiv.innerHTML = "";
            botDiv.textContent = payload.error;
            gotFirstToken = true;
          }

          if (payload.done && payload.session_id) {
            sessionId = payload.session_id;
            localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
          }
        }
      }

      if (!gotFirstToken) {
        botDiv.innerHTML = "";
        botDiv.textContent = "Sorry, I didn't get a response.";
      }
    } catch (err) {
      console.error("Chat request failed:", err);
      botDiv.innerHTML = "";
      botDiv.textContent = "Connection lost. ";
      const retryBtn = document.createElement("button");
      retryBtn.className = "mc-retry-btn";
      retryBtn.textContent = "Retry";
      retryBtn.addEventListener("click", () => {
        botDiv.remove();
        inputEl.value = text;
        sendMessage();
      });
      botDiv.appendChild(document.createElement("br"));
      botDiv.appendChild(retryBtn);
    } finally {
      sendBtn.disabled = false;
      inputEl.disabled = false;
      inputEl.focus();
    }
  }

  // ---------------------------------------------------------------
  // Clear conversation
  // ---------------------------------------------------------------

  async function clearConversation() {
    clearBtn.disabled = true;
    try {
      await fetch(`${CHAT_URL}/${sessionId}`, { method: "DELETE" });
    } catch (err) {
      console.error(
        "Failed to clear session on server (resetting locally anyway):",
        err,
      );
    }
    sessionId = newSessionId();
    messagesEl.innerHTML = "";
    addMessage(
      "Conversation cleared. Ask me about medicine availability, dosage, or alternatives.",
      "bot",
    );
    clearBtn.disabled = false;
  }

  // ---------------------------------------------------------------
  // Voice search - records via MediaRecorder, POSTs to /transcribe,
  // fills the input box for the user to review/edit (never auto-sent).
  // ---------------------------------------------------------------

  const voiceSupported =
    SHOW_VOICE_SEARCH &&
    !!navigator.mediaDevices &&
    !!navigator.mediaDevices.getUserMedia &&
    typeof MediaRecorder !== "undefined";

  if (!voiceSupported) {
    micBtn.style.display = "none";
  }

  let mediaRecorder = null;
  let audioChunks = [];
  let isRecording = false;

  async function startRecording() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.error("voice-search: mic permission denied or unavailable", err);
      showStatus("Mic access denied");
      setTimeout(() => showStatus(""), 2500);
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm")
      ? "audio/webm"
      : ""; // let the browser pick (Safari -> audio/mp4)

    mediaRecorder = mimeType
      ? new MediaRecorder(stream, { mimeType })
      : new MediaRecorder(stream);

    audioChunks = [];
    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
      sendForTranscription(blob);
    });

    mediaRecorder.start();
    isRecording = true;
    micBtn.classList.add("mc-recording");
    showStatus("Listening…");
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
    }
    isRecording = false;
    micBtn.classList.remove("mc-recording");
  }

  async function sendForTranscription(blob) {
    showStatus("Transcribing…");
    micBtn.disabled = true;

    const extension = blob.type.includes("mp4") ? "mp4" : "webm";
    const formData = new FormData();
    formData.append("audio", blob, `voice_query.${extension}`);

    try {
      const res = await fetch(TRANSCRIBE_URL, { method: "POST", body: formData });
      if (!res.ok) {
        const errBody = await res.json().catch(() => ({}));
        throw new Error(errBody.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      const text = (data.text || "").trim();

      if (!text) {
        showStatus("Didn't catch that — try again");
        setTimeout(() => showStatus(""), 2500);
        return;
      }

      const nativeSetter = Object.getOwnPropertyDescriptor(
        window.HTMLInputElement.prototype,
        "value",
      ).set;
      nativeSetter.call(inputEl, text);
      inputEl.dispatchEvent(new Event("input", { bubbles: true }));
      inputEl.focus();
      showStatus("");
    } catch (err) {
      console.error("voice-search: transcription request failed", err);
      showStatus("Voice search failed — please type instead");
      setTimeout(() => showStatus(""), 3000);
    } finally {
      micBtn.disabled = false;
    }
  }

  if (voiceSupported) {
    micBtn.addEventListener("click", () => {
      if (isRecording) {
        stopRecording();
      } else {
        startRecording();
      }
    });
  }

  // ---------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------

  bubble.addEventListener("click", async () => {
    const isOpen = win.style.display === "flex";
    win.style.display = isOpen ? "none" : "flex";

    if (!isOpen) {
      if (greetingEl) {
        greetingEl.classList.remove("mc-visible");
        localStorage.setItem(GREETING_DISMISSED_KEY, "1");
      }
      if (!historyRestored) {
        historyRestored = true;
        await restoreHistory();
        if (messagesEl.children.length === 0) {
          addMessage(
            "Hi! Ask me about medicine availability, dosage, or alternatives.",
            "bot",
          );
        }
        inputEl.focus();
      }
    }
  });

  win
    .querySelector("#mc-close")
    .addEventListener("click", () => (win.style.display = "none"));
  clearBtn.addEventListener("click", clearConversation);
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  window.addEventListener("offline", () =>
    showStatus(
      "You're offline - messages won't send until you're back online.",
    ),
  );
  window.addEventListener("online", () => showStatus(""));
})();