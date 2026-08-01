/**
 * Medical Shop Chatbot Widget
 * Drop this into any HTML page:
 *
 *   <script>
 *     window.MED_CHATBOT_API_URL = "https://your-backend.onrender.com/chat";
 *   </script>
 *   <script src="chatbot-widget.js"></script>
 *
 * That's it — a floating chat bubble appears in the bottom-right corner.
 * The backend must expose:
 *   POST   {API_URL}                  (SSE stream: "data: {token|error|done}\n\n")
 *   GET    {API_URL}/{session_id}/history
 *   DELETE {API_URL}/{session_id}
 */
(function () {
  const CHAT_URL = window.MED_CHATBOT_API_URL || "http://localhost:8000/chat";
  const SESSION_STORAGE_KEY = "medsmitra_session_id";

  // ---------------------------------------------------------------
  // Session id — persisted in localStorage so a page reload resumes
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
      position: fixed; bottom: 20px; right: 20px; width: 60px; height: 60px;
      border-radius: 50%; background: #0f766e; color: #fff; display: flex;
      align-items: center; justify-content: center; cursor: pointer;
      box-shadow: 0 4px 14px rgba(0,0,0,0.25); font-size: 26px; z-index: 999999;
      border: none; transition: transform 0.15s ease;
    }
    #mc-bubble:hover { transform: scale(1.06); }
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
    #mc-input-row { display: flex; border-top: 1px solid #ddd; }
    #mc-input { flex: 1; border: none; padding: 12px; font-size: 14px; outline: none; }
    #mc-input:disabled { background: #f3f3f3; }
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
  bubble.innerHTML = "💬";
  document.body.appendChild(bubble);

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
    <div class="mc-disclaimer">Informational only — please confirm with a pharmacist.</div>
    <div id="mc-input-row">
      <input id="mc-input" type="text" placeholder="Ask about a medicine..." />
      <button id="mc-send">Send</button>
    </div>
  `;
  document.body.appendChild(win);

  const messagesEl = win.querySelector("#mc-messages");
  const inputEl = win.querySelector("#mc-input");
  const sendBtn = win.querySelector("#mc-send");
  const clearBtn = win.querySelector("#mc-clear");
  const statusEl = win.querySelector("#mc-status");

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
    div.innerHTML = '<span class="mc-dot"></span><span class="mc-dot"></span><span class="mc-dot"></span>';
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
  // Send message — SSE streaming with reconnect/retry handling
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
      if (!res.ok || !res.body) throw new Error("Request failed: " + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const parts = buffer.split("\n\n");
        buffer = parts.pop(); // last part may be incomplete — keep for next chunk

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
      console.error("Failed to clear session on server (resetting locally anyway):", err);
    }
    sessionId = newSessionId();
    messagesEl.innerHTML = "";
    addMessage("Conversation cleared. Ask me about medicine availability, dosage, or alternatives.", "bot");
    clearBtn.disabled = false;
  }

  // ---------------------------------------------------------------
  // Wiring
  // ---------------------------------------------------------------

  bubble.addEventListener("click", async () => {
    const isOpen = win.style.display === "flex";
    win.style.display = isOpen ? "none" : "flex";

    if (!isOpen && !historyRestored) {
      historyRestored = true;
      await restoreHistory();
      if (messagesEl.children.length === 0) {
        addMessage("Hi! Ask me about medicine availability, dosage, or alternatives.", "bot");
      }
      inputEl.focus();
    }
  });

  win.querySelector("#mc-close").addEventListener("click", () => (win.style.display = "none"));
  clearBtn.addEventListener("click", clearConversation);
  sendBtn.addEventListener("click", sendMessage);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendMessage();
  });

  window.addEventListener("offline", () => showStatus("You're offline — messages won't send until you're back online."));
  window.addEventListener("online", () => showStatus(""));
})();