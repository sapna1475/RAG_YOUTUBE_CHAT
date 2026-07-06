const BACKEND_URL = "https://youtube-rag-backend-mvqq.onrender.com";

const videoLabel = document.getElementById("video-label");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("ask-btn");
const answerBox = document.getElementById("answer-box");
const errorMsg = document.getElementById("error-msg");

let currentVideoId = null;

// ── Detect video ID when popup opens ─────────────────────────────
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
  const tab = tabs[0];

  if (!tab.url || !tab.url.includes("youtube.com/watch")) {
    videoLabel.textContent = "Open a YouTube video first";
    askBtn.disabled = true;
    return;
  }

  chrome.tabs.sendMessage(tab.id, { type: "GET_VIDEO_ID" }, (response) => {
    if (chrome.runtime.lastError || !response || !response.videoId) {
      videoLabel.textContent = "Reload the YouTube page and try again";
      askBtn.disabled = true;
      return;
    }

    currentVideoId = response.videoId;
    videoLabel.textContent = `Video: ${currentVideoId}`;
  });
});

// ── Ask button ────────────────────────────────────────────────────
askBtn.addEventListener("click", async () => {
  const question = questionInput.value.trim();

  if (!question) {
    showError("Please type a question first.");
    return;
  }

  if (!currentVideoId) {
    showError("No video detected. Open a YouTube video and try again.");
    return;
  }

  // Reset UI
  askBtn.disabled = true;
  askBtn.textContent = "Thinking...";
  hideError();
  answerBox.style.display = "block";
  answerBox.textContent = "";

  try {
    // Check backend is running
    const healthCheck = await fetch(`${BACKEND_URL}/health`).catch(() => null);
    if (!healthCheck || !healthCheck.ok) {
      showError("Backend not running. Start with: uvicorn main:app --port 8000");
      return;
    }

    // Send request to /ask endpoint
    const response = await fetch(`${BACKEND_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: currentVideoId,
        question: question
      })
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      showError(err.detail || "Something went wrong.");
      return;
    }

    const data = await response.json();
    answerBox.textContent = data.answer;

  } catch (error) {
    showError("Could not connect to backend.");
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Ask";
  }
});

// Enter to submit, Shift+Enter for new line
questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    askBtn.click();
  }
});

// ── Helpers ───────────────────────────────────────────────────────
function showError(msg) {
  errorMsg.style.display = "block";
  errorMsg.textContent = msg;
}

function hideError() {
  errorMsg.style.display = "none";
  errorMsg.textContent = "";
}