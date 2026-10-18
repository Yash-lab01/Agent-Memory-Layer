/**
 * Agent Memory Layer - Interactive Demo Client
 * Zero-dependency reactive client managing chat state, session switching, and live memory inspection.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const inputUserId = document.getElementById("input-user-id");
  const selectSession = document.getElementById("select-session-id");
  const memoryToggle = document.getElementById("memory-toggle");
  const toggleStateText = document.getElementById("toggle-state-text");
  const badgeChatMode = document.getElementById("badge-chat-mode");
  const badgeInspectorUser = document.getElementById("badge-inspector-user");

  // Metrics
  const valActiveCount = document.getElementById("val-active-count");
  const valSupersededCount = document.getElementById("val-superseded-count");
  const valPromptTokens = document.getElementById("val-prompt-tokens");
  const valLatency = document.getElementById("val-latency");

  // Chat
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const btnClearChat = document.getElementById("btn-clear-chat");

  // Inspector
  const btnRefreshMemories = document.getElementById("btn-refresh-memories");
  const btnResetStore = document.getElementById("btn-reset-store");
  const tabActive = document.getElementById("tab-active");
  const tabSuperseded = document.getElementById("tab-superseded");
  const tabActiveCount = document.getElementById("tab-active-count");
  const tabSupersededCount = document.getElementById("tab-superseded-count");
  const paneActive = document.getElementById("pane-active");
  const paneSuperseded = document.getElementById("pane-superseded");
  const activeMemoriesList = document.getElementById("active-memories-list");
  const supersededMemoriesList = document.getElementById("superseded-memories-list");

  // Scenario Buttons
  const scenarioButtons = document.querySelectorAll(".btn-pill");

  // ---------------------------------------------------------------------------
  // State Helpers
  // ---------------------------------------------------------------------------

  function getUserId() {
    return (inputUserId.value || "demo_user").trim();
  }

  function getSessionId() {
    return selectSession.value || "session_1";
  }

  function isMemoryEnabled() {
    return memoryToggle.checked;
  }

  // ---------------------------------------------------------------------------
  // Memory Toggle Event Handling
  // ---------------------------------------------------------------------------

  memoryToggle.addEventListener("change", () => {
    const enabled = isMemoryEnabled();
    if (enabled) {
      toggleStateText.textContent = "Memory ON";
      toggleStateText.style.color = "var(--accent-emerald)";
      badgeChatMode.textContent = "Memory Augmented";
      badgeChatMode.className = "mode-badge badge-memory-on";
    } else {
      toggleStateText.textContent = "Memory OFF";
      toggleStateText.style.color = "var(--accent-amber)";
      badgeChatMode.textContent = "Stateless Baseline";
      badgeChatMode.className = "mode-badge badge-memory-off";
    }
  });

  inputUserId.addEventListener("change", () => {
    badgeInspectorUser.textContent = getUserId();
    fetchMemories();
  });

  // ---------------------------------------------------------------------------
  // Inspector Tab Switching
  // ---------------------------------------------------------------------------

  tabActive.addEventListener("click", () => {
    tabActive.classList.add("active");
    tabSuperseded.classList.remove("active");
    paneActive.classList.add("active");
    paneSuperseded.classList.remove("active");
  });

  tabSuperseded.addEventListener("click", () => {
    tabSuperseded.classList.add("active");
    tabActive.classList.remove("active");
    paneSuperseded.classList.add("active");
    paneActive.classList.remove("active");
  });

  // ---------------------------------------------------------------------------
  // Quick Scenario Buttons
  // ---------------------------------------------------------------------------

  scenarioButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      const session = btn.getAttribute("data-session");
      if (session) {
        selectSession.value = session;
      }
      chatInput.value = text;
      chatInput.focus();
    });
  });

  // ---------------------------------------------------------------------------
  // Chat Communication
  // ---------------------------------------------------------------------------

  btnClearChat.addEventListener("click", () => {
    chatMessages.innerHTML = `
      <div class="system-welcome-card">
        <div class="welcome-header">
          <span class="welcome-icon">💡</span>
          <h3>Chat Cleared</h3>
        </div>
        <p>Chat window cleared. Your persistent memories remain safely stored in the Memory Inspector!</p>
      </div>
    `;
  });

  btnSend.addEventListener("click", handleSend);
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  async function handleSend() {
    const message = chatInput.value.trim();
    if (!message) return;

    chatInput.value = "";
    btnSend.disabled = true;

    // Render User Bubble
    appendMessageBubble("user", message, getSessionId());

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: getUserId(),
          session_id: getSessionId(),
          message: message,
          memory_enabled: isMemoryEnabled(),
        }),
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const data = await response.json();

      // Update metrics ribbon
      valPromptTokens.textContent = data.prompt_tokens;
      valLatency.textContent = `${data.latency_ms} ms`;

      // Render Assistant Bubble
      appendAssistantBubble(data);

      // Refresh Memory Inspector
      fetchMemories();
    } catch (err) {
      appendMessageBubble("assistant", `⚠️ Error: Could not connect to backend service (${err.message})`, getSessionId());
    } finally {
      btnSend.disabled = false;
      chatInput.focus();
    }
  }

  function appendMessageBubble(role, text, session) {
    const bubble = document.createElement("div");
    bubble.className = `message-bubble ${role}`;
    bubble.innerHTML = `
      <div class="bubble-content">${escapeHtml(text)}</div>
      <div class="bubble-meta">
        <span>${escapeHtml(session)}</span>
        <span>•</span>
        <span>${new Date().toLocaleTimeString()}</span>
      </div>
    `;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAssistantBubble(data) {
    const bubble = document.createElement("div");
    bubble.className = "message-bubble assistant";

    let contextSection = "";
    if (data.memory_enabled && data.retrieved_memories && data.retrieved_memories.length > 0) {
      const factList = data.retrieved_memories
        .map((m) => `• [${escapeHtml(m.entity || "Fact")}] ${escapeHtml(m.content)} (score: ${m.score})`)
        .join("<br/>");

      contextSection = `
        <div class="context-injected-pill">
          <div class="context-pill-header">
            <span>⚡ Memory Context Injected (${data.retrieved_memories.length} facts)</span>
            <span>${data.prompt_tokens} tokens</span>
          </div>
          <div class="context-fact-snippet">${factList}</div>
        </div>
      `;
    } else if (!data.memory_enabled) {
      contextSection = `
        <div class="context-injected-pill" style="border-color: rgba(245, 158, 11, 0.3);">
          <div class="context-pill-header" style="color: var(--accent-amber);">
            <span>⚠️ Memory OFF (Stateless baseline)</span>
            <span>${data.prompt_tokens} tokens</span>
          </div>
          <div class="context-fact-snippet" style="color: #94a3b8;">Zero prior context injected. Model answers without durable memory.</div>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-content">
        ${escapeHtml(data.reply)}
        ${contextSection}
      </div>
      <div class="bubble-meta">
        <span>${escapeHtml(getSessionId())}</span>
        <span>•</span>
        <span>${data.latency_ms} ms</span>
      </div>
    `;
    chatMessages.appendChild(bubble);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // ---------------------------------------------------------------------------
  // Memory Inspector Fetching & Rendering
  // ---------------------------------------------------------------------------

  btnRefreshMemories.addEventListener("click", fetchMemories);

  btnResetStore.addEventListener("click", async () => {
    if (!confirm(`Are you sure you want to purge all memories for user '${getUserId()}'?`)) {
      return;
    }
    try {
      const res = await fetch(`/api/reset?user_id=${encodeURIComponent(getUserId())}`, {
        method: "POST",
      });
      if (res.ok) {
        fetchMemories();
      }
    } catch (e) {
      alert(`Reset error: ${e.message}`);
    }
  });

  async function fetchMemories() {
    try {
      const res = await fetch(`/api/memories?user_id=${encodeURIComponent(getUserId())}`);
      if (!res.ok) return;

      const memories = await res.json();
      const active = memories.filter((m) => m.status === "active");
      const superseded = memories.filter((m) => m.status === "superseded");

      // Update badges & metrics
      valActiveCount.textContent = active.length;
      valSupersededCount.textContent = superseded.length;
      tabActiveCount.textContent = active.length;
      tabSupersededCount.textContent = superseded.length;

      renderActiveMemories(active);
      renderSupersededMemories(superseded);
    } catch (err) {
      console.error("Failed to fetch memories:", err);
    }
  }

  function renderActiveMemories(list) {
    if (list.length === 0) {
      activeMemoriesList.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">📂</span>
          <p>No active memories extracted yet.</p>
          <span class="empty-hint">Send a chat message to extract facts!</span>
        </div>
      `;
      return;
    }

    activeMemoriesList.innerHTML = list
      .map((m) => {
        const entityClass = `entity-${(m.entity || "general").toLowerCase()}`;
        return `
          <div class="memory-card card-active" id="memory-card-${m.id}">
            <div class="card-top-row">
              <span class="entity-pill ${entityClass}">${escapeHtml(m.entity || "General")}</span>
              <span class="status-badge badge-active">Active</span>
            </div>
            <div class="memory-fact-text">${escapeHtml(m.content)}</div>
            <div class="card-footer-row">
              <div class="card-metrics">
                <span>Imp: ${(m.importance * 100).toFixed(0)}%</span>
                <span>•</span>
                <span>Hits: ${m.access_count}</span>
              </div>
              <button class="btn-card-forget" onclick="window.forgetMemory('${m.id}')" title="Forget this memory">
                ✕ Forget
              </button>
            </div>
          </div>
        `;
      })
      .join("");
  }

  function renderSupersededMemories(list) {
    if (list.length === 0) {
      supersededMemoriesList.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">🛡️</span>
          <p>No superseded facts yet.</p>
          <span class="empty-hint">Update a previous statement (e.g. Seattle &rarr; Austin) to observe supersession!</span>
        </div>
      `;
      return;
    }

    supersededMemoriesList.innerHTML = list
      .map((m) => {
        const entityClass = `entity-${(m.entity || "general").toLowerCase()}`;
        const supersededTarget = m.superseded_by ? `Superseded by #${m.superseded_by.slice(0, 8)}` : "Superseded by newer update";
        return `
          <div class="memory-card card-superseded" id="memory-card-${m.id}">
            <div class="card-top-row">
              <span class="entity-pill ${entityClass}">${escapeHtml(m.entity || "General")}</span>
              <span class="status-badge badge-superseded">Superseded</span>
            </div>
            <div class="memory-fact-text">${escapeHtml(m.content)}</div>
            <div class="supersession-provenance">
              <span>🔗 ${escapeHtml(supersededTarget)}</span>
            </div>
            <div class="card-footer-row">
              <div class="card-metrics">
                <span>Archived</span>
              </div>
              <button class="btn-card-forget" onclick="window.forgetMemory('${m.id}')" title="Permanently delete">
                ✕ Delete
              </button>
            </div>
          </div>
        `;
      })
      .join("");
  }

  // Global window handler for memory deletion
  window.forgetMemory = async function (id) {
    try {
      const res = await fetch(`/api/memories/${encodeURIComponent(id)}?user_id=${encodeURIComponent(getUserId())}`, {
        method: "DELETE",
      });
      if (res.ok) {
        fetchMemories();
      }
    } catch (err) {
      alert(`Could not delete memory: ${err.message}`);
    }
  };

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // Initial load
  fetchMemories();
});
