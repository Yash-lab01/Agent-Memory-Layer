/**
 * Agent Memory Layer - Interactive Demo Client & Knowledge Graph Visualizer
 * Zero-dependency reactive client managing chat state, session switching,
 * live memory inspection, and interactive Canvas knowledge graph visualization.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Global Controls & Navigation
  const inputUserId = document.getElementById("input-user-id");
  const selectSession = document.getElementById("select-session-id");
  const memoryToggle = document.getElementById("memory-toggle");
  const toggleStateText = document.getElementById("toggle-state-text");
  const badgeChatMode = document.getElementById("badge-chat-mode");
  const badgeInspectorUser = document.getElementById("badge-inspector-user");

  // Metrics Ribbon Elements
  const valActiveCount = document.getElementById("val-active-count");
  const valSupersededCount = document.getElementById("val-superseded-count");
  const valGraphRelations = document.getElementById("val-graph-relations");
  const valPromptTokens = document.getElementById("val-prompt-tokens");
  const valLatency = document.getElementById("val-latency");

  // Chat Interface Elements
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const btnSend = document.getElementById("btn-send");
  const btnClearChat = document.getElementById("btn-clear-chat");
  const scenarioButtons = document.querySelectorAll(".btn-pill");

  // Inspector Elements (Tabs & Panes)
  const btnRefreshMemories = document.getElementById("btn-refresh-memories");
  const btnResetStore = document.getElementById("btn-reset-store");
  const tabActive = document.getElementById("tab-active");
  const tabSuperseded = document.getElementById("tab-superseded");
  const tabGraph = document.getElementById("tab-graph");
  const tabWanderai = document.getElementById("tab-wanderai");
  const tabActiveCount = document.getElementById("tab-active-count");
  const tabSupersededCount = document.getElementById("tab-superseded-count");
  const tabGraphCount = document.getElementById("tab-graph-count");
  const tabWanderaiCount = document.getElementById("tab-wanderai-count");
  const paneActive = document.getElementById("pane-active");
  const paneSuperseded = document.getElementById("pane-superseded");
  const paneGraph = document.getElementById("pane-graph");
  const paneWanderai = document.getElementById("pane-wanderai");
  const activeMemoriesList = document.getElementById("active-memories-list");
  const supersededMemoriesList = document.getElementById("superseded-memories-list");

  // WanderAI Elements
  const wanderaiProfileUser = document.getElementById("wanderai-profile-user");
  const btnWanderaiSeed = document.getElementById("btn-wanderai-seed");
  const profileDietary = document.getElementById("profile-dietary");
  const profileLoyalty = document.getElementById("profile-loyalty");
  const profileBudget = document.getElementById("profile-budget");
  const profileInterests = document.getElementById("profile-interests");
  const wanderaiActiveList = document.getElementById("wanderai-active-list");
  const wanderaiSupersededList = document.getElementById("wanderai-superseded-list");

  // Knowledge Graph Elements
  const graphCanvas = document.getElementById("graph-canvas");
  const graphContainer = document.getElementById("graph-container");
  const btnFitGraph = document.getElementById("btn-fit-graph");
  const btnClearGraph = document.getElementById("btn-clear-graph");
  const btnOpenAddTriplet = document.getElementById("btn-open-add-triplet");
  const modalAddTriplet = document.getElementById("modal-add-triplet");
  const formAddTriplet = document.getElementById("form-add-triplet");
  const btnCloseTripletModal = document.getElementById("btn-close-triplet-modal");
  const btnCancelTriplet = document.getElementById("btn-cancel-triplet");
  const tripletSource = document.getElementById("triplet-source");
  const tripletRelation = document.getElementById("triplet-relation");
  const tripletTarget = document.getElementById("triplet-target");

  // Path Finder Elements
  const inputPathSource = document.getElementById("input-path-source");
  const inputPathTarget = document.getElementById("input-path-target");
  const btnFindPath = document.getElementById("btn-find-path");
  const btnClearPath = document.getElementById("btn-clear-path");
  const pathResultBanner = document.getElementById("path-result-banner");
  const graphNodesDatalist = document.getElementById("graph-nodes-datalist");

  // Node Details Elements
  const graphNodeDetails = document.getElementById("graph-node-details");
  const detailNodeName = document.getElementById("detail-node-name");
  const detailNodeType = document.getElementById("detail-node-type");
  const detailNodeRelations = document.getElementById("detail-node-relations");
  const btnCloseNodeDetails = document.getElementById("btn-close-node-details");

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
    fetchGraph();
  });

  // ---------------------------------------------------------------------------
  // Inspector Tab Switching
  // ---------------------------------------------------------------------------

  function switchTab(activeTabName) {
    [tabActive, tabSuperseded, tabGraph, tabWanderai].forEach((t) => t && t.classList.remove("active"));
    [paneActive, paneSuperseded, paneGraph, paneWanderai].forEach((p) => p && p.classList.remove("active"));

    if (activeTabName === "active") {
      tabActive.classList.add("active");
      paneActive.classList.add("active");
    } else if (activeTabName === "superseded") {
      tabSuperseded.classList.add("active");
      paneSuperseded.classList.add("active");
    } else if (activeTabName === "graph") {
      tabGraph.classList.add("active");
      paneGraph.classList.add("active");
      if (graphVisualizer) {
        setTimeout(() => {
          graphVisualizer.resize();
          graphVisualizer.wake();
        }, 30);
      }
    } else if (activeTabName === "wanderai") {
      tabWanderai.classList.add("active");
      paneWanderai.classList.add("active");
      fetchWanderAIProfile();
    }
  }

  tabActive.addEventListener("click", () => switchTab("active"));
  tabSuperseded.addEventListener("click", () => switchTab("superseded"));
  tabGraph.addEventListener("click", () => switchTab("graph"));
  tabWanderai.addEventListener("click", () => switchTab("wanderai"));

  // ---------------------------------------------------------------------------
  // Quick Scenario Buttons
  // ---------------------------------------------------------------------------

  scenarioButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const text = btn.getAttribute("data-text");
      const session = btn.getAttribute("data-session");
      const user = btn.getAttribute("data-user");
      if (user) {
        inputUserId.value = user;
        badgeInspectorUser.textContent = user;
      }
      if (session) {
        selectSession.value = session;
      }
      chatInput.value = text;
      chatInput.focus();

      if (btn.classList.contains("btn-pill-wanderai")) {
        switchTab("wanderai");
      }
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
        <p>Chat window cleared. Your persistent memories and knowledge graph remain safely stored in the Memory Inspector!</p>
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

      // Refresh Memory Inspector & Graph
      fetchMemories();
      fetchGraph();
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
            <span>⚡ Episodic Memory Injected (${data.retrieved_memories.length} facts)</span>
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

    let graphSection = "";
    if (data.memory_enabled && data.graph_context && data.graph_context.trim().length > 0) {
      graphSection = `
        <div class="graph-context-pill">
          <div class="graph-pill-header">
            <span>🌐 Knowledge Graph Subgraph Injected</span>
          </div>
          <div class="graph-fact-snippet">${escapeHtml(data.graph_context).replace(/\n/g, "<br/>")}</div>
        </div>
      `;
    }

    bubble.innerHTML = `
      <div class="bubble-content">
        ${escapeHtml(data.reply)}
        ${contextSection}
        ${graphSection}
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

  btnRefreshMemories.addEventListener("click", () => {
    fetchMemories();
    fetchGraph();
  });

  btnResetStore.addEventListener("click", async () => {
    if (!confirm(`Are you sure you want to purge all episodic memories for user '${getUserId()}'?`)) {
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

  // Global window handler for episodic memory deletion
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

  // ---------------------------------------------------------------------------
  // Canvas Force-Directed Knowledge Graph Visualizer
  // ---------------------------------------------------------------------------

  class ForceGraphVisualizer {
    constructor(canvas, container, options = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.container = container;
      this.nodes = [];
      this.edges = [];
      this.nodeMap = new Map();
      this.highlightedEdgeIds = new Set();
      this.highlightedNodeIds = new Set();
      this.selectedNode = null;
      this.draggedNode = null;
      this.isSimulating = false;
      this.onSelectNode = options.onSelectNode || null;

      this.initEvents();
      this.resize();
    }

    resize() {
      const rect = this.container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;

      const dpr = window.devicePixelRatio || 1;
      this.width = rect.width;
      this.height = rect.height;
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;

      this.ctx.resetTransform();
      this.ctx.scale(dpr, dpr);
      this.wake();
    }

    setData(rawNodes, rawEdges) {
      const prevPos = new Map();
      this.nodes.forEach((n) => prevPos.set(n.name, { x: n.x, y: n.y }));

      this.nodeMap.clear();
      const cx = (this.width || 400) / 2;
      const cy = (this.height || 400) / 2;

      this.nodes = rawNodes.map((n, i) => {
        const saved = prevPos.get(n.name);
        const angle = (i / Math.max(rawNodes.length, 1)) * Math.PI * 2;
        const radius = Math.min(this.width || 400, this.height || 400) * 0.32;
        const node = {
          id: n.id,
          name: n.name,
          entity_type: n.entity_type || "general",
          properties: n.properties || {},
          x: saved ? saved.x : cx + Math.cos(angle) * radius,
          y: saved ? saved.y : cy + Math.sin(angle) * radius,
          vx: 0,
          vy: 0,
          radius: 18,
        };
        this.nodeMap.set(n.name, node);
        return node;
      });

      this.edges = rawEdges
        .map((e) => ({
          id: e.id,
          sourceName: e.source,
          targetName: e.target,
          source: this.nodeMap.get(e.source),
          target: this.nodeMap.get(e.target),
          relation: e.relation,
          weight: e.weight || 1.0,
          properties: e.properties || {},
        }))
        .filter((e) => e.source && e.target);

      this.wake();
    }

    wake() {
      if (!this.isSimulating) {
        this.isSimulating = true;
        requestAnimationFrame(() => this.step());
      }
    }

    step() {
      if (!this.width || !this.height) {
        this.isSimulating = false;
        return;
      }

      const cx = this.width / 2;
      const cy = this.height / 2;
      const kRepulsion = 2400;
      const springLen = 100;
      const springK = 0.04;

      // 1. Repulsion between all node pairs
      for (let i = 0; i < this.nodes.length; i++) {
        const n1 = this.nodes[i];
        for (let j = i + 1; j < this.nodes.length; j++) {
          const n2 = this.nodes[j];
          const dx = n2.x - n1.x;
          const dy = n2.y - n1.y;
          const distSq = dx * dx + dy * dy + 1;
          const dist = Math.sqrt(distSq);
          const force = kRepulsion / distSq;
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;
          if (n1 !== this.draggedNode) {
            n1.vx -= fx;
            n1.vy -= fy;
          }
          if (n2 !== this.draggedNode) {
            n2.vx += fx;
            n2.vy += fy;
          }
        }
      }

      // 2. Spring attraction along relation edges
      for (const edge of this.edges) {
        const n1 = edge.source;
        const n2 = edge.target;
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const displacement = dist - springLen;
        const force = displacement * springK;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;
        if (n1 !== this.draggedNode) {
          n1.vx += fx;
          n1.vy += fy;
        }
        if (n2 !== this.draggedNode) {
          n2.vx -= fx;
          n2.vy -= fy;
        }
      }

      // 3. Center gravity & update positions
      let totalMovement = 0;
      for (const n of this.nodes) {
        if (n === this.draggedNode) continue;

        // Soft center gravitation
        n.vx += (cx - n.x) * 0.015;
        n.vy += (cy - n.y) * 0.015;

        // Friction damping
        n.vx *= 0.85;
        n.vy *= 0.85;

        n.x += n.vx;
        n.y += n.vy;

        // Boundary containment
        const pad = n.radius + 15;
        n.x = Math.max(pad, Math.min(this.width - pad, n.x));
        n.y = Math.max(pad, Math.min(this.height - pad, n.y));

        totalMovement += Math.abs(n.vx) + Math.abs(n.vy);
      }

      this.draw();

      if (totalMovement > 0.08 || this.draggedNode) {
        requestAnimationFrame(() => this.step());
      } else {
        this.isSimulating = false;
      }
    }

    draw() {
      const ctx = this.ctx;
      ctx.clearRect(0, 0, this.width, this.height);

      // 1. Draw directed edges
      for (const edge of this.edges) {
        const isPath = this.highlightedEdgeIds.has(edge.id);
        const isConnected =
          this.selectedNode &&
          (edge.source === this.selectedNode || edge.target === this.selectedNode);

        ctx.save();
        ctx.lineWidth = isPath ? 2.8 : isConnected ? 2.0 : 1.2;
        ctx.strokeStyle = isPath
          ? "#10b981"
          : isConnected
          ? "#818cf8"
          : "rgba(255, 255, 255, 0.22)";

        const dx = edge.target.x - edge.source.x;
        const dy = edge.target.y - edge.source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const ux = dx / dist;
        const uy = dy / dist;

        const sx = edge.source.x + ux * edge.source.radius;
        const sy = edge.source.y + uy * edge.source.radius;
        const tx = edge.target.x - ux * edge.target.radius;
        const ty = edge.target.y - uy * edge.target.radius;

        // Line
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(tx, ty);
        ctx.stroke();

        // Arrow head
        const arrowSize = isPath ? 9 : 7;
        ctx.fillStyle = ctx.strokeStyle;
        ctx.beginPath();
        ctx.moveTo(tx, ty);
        ctx.lineTo(
          tx - ux * arrowSize - uy * (arrowSize * 0.45),
          ty - uy * arrowSize + ux * (arrowSize * 0.45)
        );
        ctx.lineTo(
          tx - ux * arrowSize + uy * (arrowSize * 0.45),
          ty - uy * arrowSize - ux * (arrowSize * 0.45)
        );
        ctx.closePath();
        ctx.fill();

        // Predicate label badge
        const midX = (sx + tx) / 2;
        const midY = (sy + ty) / 2;
        ctx.font = "9px 'JetBrains Mono', monospace";
        const textMetrics = ctx.measureText(edge.relation);
        const textWidth = textMetrics.width;

        ctx.fillStyle = isPath
          ? "rgba(16, 185, 129, 0.95)"
          : isConnected
          ? "rgba(99, 102, 241, 0.9)"
          : "rgba(15, 23, 42, 0.85)";
        ctx.strokeStyle = isPath
          ? "#10b981"
          : isConnected
          ? "#818cf8"
          : "rgba(255, 255, 255, 0.15)";
        ctx.lineWidth = 1;

        ctx.beginPath();
        ctx.roundRect(midX - textWidth / 2 - 4, midY - 7, textWidth + 8, 14, 4);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(edge.relation, midX, midY);

        ctx.restore();
      }

      // 2. Draw entity vertices
      for (const node of this.nodes) {
        const isPath =
          this.highlightedNodeIds.has(node.id) || this.highlightedNodeIds.has(node.name);
        const isSelected = this.selectedNode === node;

        ctx.save();

        // Pulsing glow halo for selected or path node
        if (isPath || isSelected) {
          ctx.beginPath();
          ctx.arc(node.x, node.y, node.radius + 6, 0, Math.PI * 2);
          ctx.fillStyle = isPath
            ? "rgba(16, 185, 129, 0.3)"
            : "rgba(99, 102, 241, 0.35)";
          ctx.fill();
        }

        // Node circle gradient
        const grad = ctx.createRadialGradient(
          node.x - 3,
          node.y - 3,
          2,
          node.x,
          node.y,
          node.radius
        );
        const colors = this.getNodeColors(node.entity_type);
        grad.addColorStop(0, colors.light);
        grad.addColorStop(1, colors.dark);

        ctx.beginPath();
        ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        ctx.lineWidth = isPath ? 2.5 : isSelected ? 2.5 : 1.5;
        ctx.strokeStyle = isPath ? "#34d399" : isSelected ? "#a5b4fc" : colors.border;
        ctx.stroke();

        // Node label
        ctx.font = "600 11px Inter, sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillStyle = "#ffffff";
        ctx.shadowColor = "rgba(0, 0, 0, 0.9)";
        ctx.shadowBlur = 4;
        ctx.fillText(node.name, node.x, node.y + node.radius + 12);

        ctx.restore();
      }
    }

    getNodeColors(entityType) {
      const t = (entityType || "").toLowerCase();
      if (t.includes("person") || t.includes("user")) {
        return { light: "#38bdf8", dark: "#0284c7", border: "#7dd3fc" };
      } else if (t.includes("project")) {
        return { light: "#c084fc", dark: "#7e22ce", border: "#e9d5ff" };
      } else if (t.includes("tech") || t.includes("lang") || t.includes("tool")) {
        return { light: "#34d399", dark: "#059669", border: "#6ee7b7" };
      } else if (t.includes("location") || t.includes("city")) {
        return { light: "#fbbf24", dark: "#d97706", border: "#fde68a" };
      }
      return { light: "#818cf8", dark: "#4338ca", border: "#a5b4fc" };
    }

    initEvents() {
      const getPos = (e) => {
        const rect = this.canvas.getBoundingClientRect();
        const clientX = e.touches ? e.touches[0].clientX : e.clientX;
        const clientY = e.touches ? e.touches[0].clientY : e.clientY;
        return {
          x: clientX - rect.left,
          y: clientY - rect.top,
        };
      };

      const findNodeAt = (pos) => {
        for (let i = this.nodes.length - 1; i >= 0; i--) {
          const n = this.nodes[i];
          const dx = pos.x - n.x;
          const dy = pos.y - n.y;
          if (dx * dx + dy * dy <= (n.radius + 10) * (n.radius + 10)) {
            return n;
          }
        }
        return null;
      };

      const onDown = (e) => {
        const pos = getPos(e);
        const hit = findNodeAt(pos);
        if (hit) {
          this.draggedNode = hit;
          this.wake();
        }
      };

      const onMove = (e) => {
        if (this.draggedNode) {
          const pos = getPos(e);
          this.draggedNode.x = pos.x;
          this.draggedNode.y = pos.y;
          this.draggedNode.vx = 0;
          this.draggedNode.vy = 0;
          this.wake();
        }
      };

      const onUp = (e) => {
        if (this.draggedNode) {
          const pos = getPos(e.changedTouches ? e.changedTouches[0] : e);
          const hit = findNodeAt(pos);
          if (hit && hit === this.draggedNode) {
            this.selectNode(hit);
          }
          this.draggedNode = null;
          this.wake();
        } else {
          const pos = getPos(e.changedTouches ? e.changedTouches[0] : e);
          const hit = findNodeAt(pos);
          if (!hit) {
            this.selectNode(null);
          }
        }
      };

      this.canvas.addEventListener("mousedown", onDown);
      window.addEventListener("mousemove", onMove);
      window.addEventListener("mouseup", onUp);

      this.canvas.addEventListener("touchstart", onDown, { passive: true });
      window.addEventListener("touchmove", onMove, { passive: true });
      window.addEventListener("touchend", onUp, { passive: true });
    }

    selectNode(node) {
      this.selectedNode = node;
      this.draw();
      if (this.onSelectNode) {
        this.onSelectNode(node);
      }
    }

    highlightPath(edgesList) {
      this.highlightedEdgeIds.clear();
      this.highlightedNodeIds.clear();

      for (const e of edgesList) {
        this.highlightedEdgeIds.add(e.id);
        this.highlightedNodeIds.add(e.source);
        this.highlightedNodeIds.add(e.target);
      }
      this.draw();
    }

    clearPath() {
      this.highlightedEdgeIds.clear();
      this.highlightedNodeIds.clear();
      this.draw();
    }

    fitView() {
      if (this.nodes.length === 0) return;
      const cx = (this.width || 400) / 2;
      const cy = (this.height || 400) / 2;
      this.nodes.forEach((n, i) => {
        const angle = (i / this.nodes.length) * Math.PI * 2;
        const radius = Math.min(this.width, this.height) * 0.28;
        n.x = cx + Math.cos(angle) * radius;
        n.y = cy + Math.sin(angle) * radius;
        n.vx = 0;
        n.vy = 0;
      });
      this.wake();
    }
  }

  // Instantiate Visualizer
  const graphVisualizer = new ForceGraphVisualizer(graphCanvas, graphContainer, {
    onSelectNode: (node) => {
      if (!node) {
        graphNodeDetails.style.display = "none";
        return;
      }

      graphNodeDetails.style.display = "flex";
      detailNodeName.textContent = node.name;
      detailNodeType.textContent = node.entity_type || "Node";

      const relatedEdges = graphVisualizer.edges.filter(
        (e) => e.sourceName === node.name || e.targetName === node.name
      );

      if (relatedEdges.length === 0) {
        detailNodeRelations.innerHTML = `<span style="color: var(--text-dim);">No direct connections.</span>`;
      } else {
        detailNodeRelations.innerHTML = relatedEdges
          .map((e) => {
            const isOutgoing = e.sourceName === node.name;
            const dirIcon = isOutgoing ? "&rarr;" : "&larr;";
            const other = isOutgoing ? e.targetName : e.sourceName;
            return `
              <div class="detail-relation-row">
                <span style="color: var(--accent-primary);">${dirIcon}</span>
                <span class="relation-badge">${escapeHtml(e.relation)}</span>
                <span style="color: #f1f5f9; font-weight: 500;">${escapeHtml(other)}</span>
              </div>
            `;
          })
          .join("");
      }
    },
  });

  btnCloseNodeDetails.addEventListener("click", () => {
    graphVisualizer.selectNode(null);
  });

  btnFitGraph.addEventListener("click", () => {
    graphVisualizer.fitView();
  });

  // ---------------------------------------------------------------------------
  // Knowledge Graph API Integration
  // ---------------------------------------------------------------------------

  async function fetchGraph() {
    try {
      const res = await fetch(`/api/graph?user_id=${encodeURIComponent(getUserId())}`);
      if (!res.ok) return;

      const data = await res.json();
      valGraphRelations.textContent = data.edges.length;
      tabGraphCount.textContent = data.nodes.length;

      // Populate datalist for path finder
      graphNodesDatalist.innerHTML = data.nodes
        .map((n) => `<option value="${escapeHtml(n.name)}">`)
        .join("");

      graphVisualizer.setData(data.nodes, data.edges);
    } catch (err) {
      console.error("Failed to fetch knowledge graph:", err);
    }
  }

  // Add Triplet Modal Handlers
  btnOpenAddTriplet.addEventListener("click", () => {
    modalAddTriplet.style.display = "flex";
    tripletSource.focus();
  });

  function closeTripletModal() {
    modalAddTriplet.style.display = "none";
    formAddTriplet.reset();
  }

  btnCloseTripletModal.addEventListener("click", closeTripletModal);
  btnCancelTriplet.addEventListener("click", closeTripletModal);

  formAddTriplet.addEventListener("submit", async (e) => {
    e.preventDefault();
    const source = tripletSource.value.trim();
    const relation = tripletRelation.value.trim();
    const target = tripletTarget.value.trim();

    if (!source || !relation || !target) return;

    try {
      const res = await fetch("/api/graph/triplets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: getUserId(),
          source: source,
          relation: relation,
          target: target,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      closeTripletModal();
      fetchGraph();
    } catch (err) {
      alert(`Could not add triplet: ${err.message}`);
    }
  });

  // Multi-Hop Path Finder
  btnFindPath.addEventListener("click", async () => {
    const source = inputPathSource.value.trim();
    const target = inputPathTarget.value.trim();

    if (!source || !target) {
      alert("Please specify both source and target entities.");
      return;
    }

    try {
      const url = `/api/graph/paths?user_id=${encodeURIComponent(getUserId())}&source=${encodeURIComponent(source)}&target=${encodeURIComponent(target)}&max_depth=4`;
      const res = await fetch(url);
      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const paths = await res.json();
      if (paths.length > 0) {
        const pathEdges = paths[0].path;
        graphVisualizer.highlightPath(pathEdges);

        const breadcrumbs = pathEdges
          .map(
            (e) =>
              `<strong>${escapeHtml(e.source)}</strong> <span class="path-arrow">&rarr;</span> <span class="relation-badge">${escapeHtml(e.relation)}</span> <span class="path-arrow">&rarr;</span> <strong>${escapeHtml(e.target)}</strong>`
          )
          .join(" ");

        pathResultBanner.innerHTML = `<span>⚡ Path Discovered:</span> ${breadcrumbs}`;
        pathResultBanner.className = "path-result-banner";
        pathResultBanner.style.display = "flex";
      } else {
        graphVisualizer.clearPath();
        pathResultBanner.innerHTML = `<span>⚠️ No path found between "${escapeHtml(source)}" and "${escapeHtml(target)}" within 4 hops.</span>`;
        pathResultBanner.className = "path-result-banner no-path";
        pathResultBanner.style.display = "flex";
      }
    } catch (err) {
      alert(`Path finding error: ${err.message}`);
    }
  });

  btnClearPath.addEventListener("click", () => {
    inputPathSource.value = "";
    inputPathTarget.value = "";
    pathResultBanner.style.display = "none";
    graphVisualizer.clearPath();
  });

  // Reset Graph
  btnClearGraph.addEventListener("click", async () => {
    if (!confirm(`Are you sure you want to purge the knowledge graph for '${getUserId()}'?`)) {
      return;
    }
    try {
      const res = await fetch(`/api/graph?user_id=${encodeURIComponent(getUserId())}`, {
        method: "DELETE",
      });
      if (res.ok) {
        graphVisualizer.clearPath();
        fetchGraph();
      }
    } catch (err) {
      alert(`Could not reset knowledge graph: ${err.message}`);
    }
  });

  // ---------------------------------------------------------------------------
  // WanderAI Traveler Profile Integration
  // ---------------------------------------------------------------------------

  async function fetchWanderAIProfile() {
    try {
      const res = await fetch(`/api/wanderai/profile?user_id=${encodeURIComponent(getUserId())}`);
      if (!res.ok) return;

      const prof = await res.json();
      if (wanderaiProfileUser) wanderaiProfileUser.textContent = prof.user_id;
      if (tabWanderaiCount) tabWanderaiCount.textContent = prof.active_preferences.length;

      if (profileDietary) {
        profileDietary.textContent = prof.dietary.length
          ? prof.dietary[0].replace("Traveler maintains a strict ", "").replace(" diet.", "")
          : "None stated";
      }
      if (profileLoyalty) {
        profileLoyalty.textContent = prof.airline_loyalty.length
          ? prof.airline_loyalty[0].replace("Traveler holds ", "").replace(" loyalty status.", "")
          : "None active";
      }
      if (profileBudget) {
        profileBudget.textContent = prof.budget_tier.length
          ? prof.budget_tier[0].replace("Traveler accommodation budget is ", "").replace(".", "")
          : "Flexible";
      }
      if (profileInterests) {
        profileInterests.textContent = prof.interests.length
          ? prof.interests[0].replace("Traveler loves ", "").replace(".", "")
          : "General";
      }

      renderWanderAILists(prof.active_preferences, prof.superseded_preferences);
    } catch (e) {
      console.error("Failed to fetch WanderAI profile:", e);
    }
  }

  function renderWanderAILists(active, superseded) {
    if (!wanderaiActiveList || !wanderaiSupersededList) return;

    if (!active.length) {
      wanderaiActiveList.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">🎒</span>
          <p>No traveler preferences active.</p>
          <span class="empty-hint">Click a WanderAI preset scenario above to initialize!</span>
        </div>
      `;
    } else {
      wanderaiActiveList.innerHTML = active
        .map(
          (m) => `
          <div class="memory-card card-active">
            <div class="card-top-row">
              <span class="entity-pill entity-${(m.entity || "general").toLowerCase()}">${escapeHtml(m.entity || "Preference")}</span>
              <span class="status-badge badge-active">Active</span>
            </div>
            <div class="memory-fact-text">${escapeHtml(m.content)}</div>
            <div class="card-footer-row">
              <span class="card-metrics">Imp: ${(m.importance * 100).toFixed(0)}%</span>
              <button class="btn-card-forget" onclick="window.forgetMemory('${m.id}')">✕ Forget</button>
            </div>
          </div>
        `
        )
        .join("");
    }

    if (!superseded.length) {
      wanderaiSupersededList.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">🛡️</span>
          <p>No superseded travel preferences yet.</p>
          <span class="empty-hint">Update loyalty or budget to observe non-destructive supersession!</span>
        </div>
      `;
    } else {
      wanderaiSupersededList.innerHTML = superseded
        .map(
          (m) => `
          <div class="memory-card card-superseded">
            <div class="card-top-row">
              <span class="entity-pill entity-${(m.entity || "general").toLowerCase()}">${escapeHtml(m.entity || "Preference")}</span>
              <span class="status-badge badge-superseded">Superseded</span>
            </div>
            <div class="memory-fact-text">${escapeHtml(m.content)}</div>
            <div class="supersession-provenance">
              <span>🔗 Superseded by #${escapeHtml((m.superseded_by || "").slice(0, 8))}</span>
            </div>
          </div>
        `
        )
        .join("");
    }
  }

  if (btnWanderaiSeed) {
    btnWanderaiSeed.addEventListener("click", async () => {
      try {
        const res = await fetch(`/api/wanderai/seed?user_id=${encodeURIComponent(getUserId())}`, {
          method: "POST",
        });
        if (res.ok) {
          fetchWanderAIProfile();
          fetchGraph();
        }
      } catch (err) {
        alert("Error seeding destination graph: " + err.message);
      }
    });
  }

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
  fetchGraph();
  fetchWanderAIProfile();
});
